# from email import message  # Unused import
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from .config import Config
from .llms import get_llm
from .prompts import Prompts
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.types import RetryPolicy

class AgentState(TypedDict):
    platform: str
    image: str
    image_type: str
    image_info: str
    style: str
    space: str
    emotion: str
    text: str
    improvement_suggestions: str | None
    legal: bool
    not_legal_info: str | None
    approved: bool

class LegalCheck(TypedDict):
    legal: Literal[True, False]
    info: str | None


llm_chat, llm_embedding, vl_model = get_llm(Config.LLM_TYPE)
structured_vllm = vl_model.with_structured_output(LegalCheck)

# Constants
IMAGE_URL_FORMAT = 'data:{};base64,{}'

async def extract_image_context_node(state: AgentState) -> dict[str, str]:
    """
    Extracts image information using a language model.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict[str, str]: A dictionary containing the extracted image information.
    """
    image = state["image"]
    image_type = state["image_type"]
    prompt = Prompts.EXTRACT_IMAGE_CONTEXT_PROMPT
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL_FORMAT.format(image_type, image)}},
                {"type": "text", "text": ""},
            ],
        },
    ]
    try:
        image_info = await vl_model.ainvoke(messages)
        return {"image_info": image_info.content_blocks[0].get('text', "")}
    except Exception as e:
        print(f"Error extracting image info: {e}")
        return {"image_info": ""}

async def check_image_safety_node(state: AgentState) -> Command[Literal["generate_text_node", "handle_non_compliant_item_node"]]:
    """
    Checks the validity of an image using a language model.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Command[Literal["generate_text_node", "handle_non_compliant_item_node"]]: A command indicating the next node.
    """
    image = state["image"]
    image_type = state["image_type"]
    image_info = state["image_info"]
    prompt = Prompts.CHECK_IMAGE_SAFETY_PROMPT
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL_FORMAT.format(image_type, image)}},
                {"type": "text", "text": image_info},
            ],
        },
    ]
    try:
        result = await structured_vllm.ainvoke(messages)
        legal = result['legal']
        return Command(goto='handle_non_compliant_item_node') if not legal else Command(goto='generate_text_node')
    except Exception as e:
        print(f"Error checking image validity: {e}")
        return Command(goto='handle_non_compliant_item_node')


async def generate_text_node(state: AgentState) -> dict[str, str]:
    """
    Generates text based on the image information and other state parameters.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict[str, str]: A dictionary containing the generated text.
    """
    image = state["image"]
    image_type = state["image_type"]
    image_info = state["image_info"]
    prompt = Prompts.GENERATE_TEXT_PROMPT

    if state.get('platform'):
        prompt += f"适配{state.get('platform')}平台风格。"
    if state.get('emotion'):
        prompt += f"情感为：{state.get('emotion')}。"
    if state.get("style"):
        prompt += f"文案风格为：{state.get('style')}。"
    if state.get("space"):
        prompt += f"文案篇幅为：{state.get('space')}。"
    if state.get("improvement_suggestions"):
        prompt += f"对之前版本文案：{state.get("text")}的修改建议：{state.get("improvement_suggestions")}。"
    if state.get("not_legal_info"):
        prompt += f"不可包含{state.get('not_legal_info')}等不合规信息。"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL_FORMAT.format(image_type, image)}},
                {"type": "text", "text": image_info},
            ],
        },
    ]
    try:
        result = await vl_model.ainvoke(messages)
        return {"text": result.content_blocks[0].get('text', "")}
    except Exception as e:
        print(f"Error generating text: {e}")
        return {"text": ""}
    

async def validate_text_compliance_node(state: AgentState) -> Command[Literal["evaluate_image_text_relevance_node", "generate_text_node"]]:
    """
    Checks the validity of the generated text using a language model.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Command[Literal["evaluate_image_text_relevance_node", "generate_text_node"]]: A command indicating the next node.
    """
    text = state["text"]
    prompt = Prompts.VALIDATE_TEXT_COMLIANCE_PROMPT
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    try:
        result = await structured_vllm.ainvoke(messages)
        legal = result['legal']
        return Command(update={"not_legal_info": result['info']}, goto='generate_text_node') if not legal else Command(goto='evaluate_image_text_relevance_node')
    except Exception as e:
        print(f"Error checking text validity: {e}")
        return Command(update={"not_legal_info": ""}, goto='generate_text_node')

async def evaluate_image_text_relevance_node(state: AgentState) -> Command[Literal["generate_text_node", "human_review_node"]]:
    """
    Checks the accuracy of the generated text using a language model.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Command[Literal["generate_text_node", "human_review_node"]]: A command indicating the next node.
    """
    image = state["image"]
    image_type = state["image_type"]
    text = state["text"]
    prompt = Prompts.EVALUATE_IMAGE_TEXT_RELEVANCE_PROMPT
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL_FORMAT.format(image_type, image)}},
                {"type": "text", "text": text},
            ],
        },
    ]
    try:
        result = await structured_vllm.ainvoke(messages)
        legal = result['legal']
        return Command(goto='generate_text_node') if not legal else Command(goto='human_review_node')
    except Exception as e:
        print(f"Error checking text accuracy: {e}")
        return Command(goto='generate_text_node')


async def human_review_node(state: AgentState) -> Command[Literal["approve_node", "generate_text_node"]]:
    """
    Allows a human to check and approve or refuse the generated text.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Command[Literal["approve_node", "refuse_node", "generate_text_node"]]: A command indicating the next node.
    """
    text = state["text"]
    result = interrupt(text)
    decision = result['decision']
    if decision == 'approve':
        return Command(goto='approve_node')
    else:
        return Command(update={"improvement_suggestions": result['info']}, goto='generate_text_node')

async def approve_node(state: AgentState) -> dict[str, bool]:
    """
    Approves the generated text.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict[str, bool]: A dictionary indicating the approval status.
    """
    return {"approved": True}

async def handle_non_compliant_item_node(state: AgentState) -> dict[str, bool]:
    """
    Indicates that the image is not valid.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        dict[str, bool]: A dictionary indicating the validity status.
    """
    return {"legal": False}



async def create_graph(checkpointer, store, **kwargs) -> StateGraph:
    """
    Creates and compiles a state graph for the creative content generation process.

    Args:
        checkpointer: The checkpointer to use.
        store: The store to use.
        **kwargs: Additional keyword arguments.

    Returns:
        StateGraph: The compiled state graph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("extract_image_context_node", extract_image_context_node, retry_policy=RetryPolicy(max_attempts=Config.NODE_ATTEMPTS))
    graph.add_node("check_image_safety_node", check_image_safety_node, retry_policy=RetryPolicy(max_attempts=Config.NODE_ATTEMPTS))
    graph.add_node("generate_text_node", generate_text_node, retry_policy=RetryPolicy(max_attempts=Config.NODE_ATTEMPTS))
    graph.add_node("validate_text_compliance_node", validate_text_compliance_node, retry_policy=RetryPolicy(max_attempts=Config.NODE_ATTEMPTS))
    graph.add_node("evaluate_image_text_relevance_node", evaluate_image_text_relevance_node, retry_policy=RetryPolicy(max_attempts=Config.NODE_ATTEMPTS))
    graph.add_node("human_review_node", human_review_node)
    graph.add_node("approve_node", approve_node)
    graph.add_node("handle_non_compliant_item_node", handle_non_compliant_item_node)

    graph.add_edge(START, "extract_image_context_node")
    graph.add_edge("extract_image_context_node", "check_image_safety_node")
    graph.add_edge("generate_text_node", "validate_text_compliance_node")
    graph.add_edge("approve_node", END)
    graph.add_edge("handle_non_compliant_item_node", END)

    return graph.compile(checkpointer=checkpointer)
