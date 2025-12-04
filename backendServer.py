import io
import base64
import logging
import uvicorn
from PIL import Image
from pydantic import BaseModel
from typing import Optional
from utils.config import Config
from utils.graph import create_graph
from langgraph.types import Command
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from fastapi import FastAPI, HTTPException, UploadFile, File, Form


# 定义数据模型 客户端发起的运行智能体的请求数据
class AgentRequest(BaseModel):
    """Client request model to run the agent."""
    session_id: str # 会话唯一标识
    style: str  # Writing style
    space: str  # Length of the copy
    emotion: str  # Mood of the copy
    platform: str  # Platform for copy deployment
    image: Optional[UploadFile] = File(None)  # Optional image upload

# 定义数据模型 客户端发起的恢复智能体运行的中断反馈请求数据
class InterruptRequest(BaseModel):
    """Client request model to resume the agent run with user feedback."""
    session_id: str # 会话唯一标识
    # decision: str  # Operation type: "approve"|"reject"|"edit"
    info: Optional[str] = None  # Edit content (only required for "edit")

# 定义数据模型 后端接口的返回值
class AgentResponse(BaseModel):
    """Response model for the agent endpoints."""
    review_content: str
    status: str

# 生命周期函数 app应用初始化函数
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 创建数据库连接池 动态连接池根据负载调整连接池大小
        # async with AsyncPostgresSaver.from_conn_string(Config.DB_URI) as pool:
        async with AsyncConnectionPool(
                conninfo=Config.DB_URI,
                min_size=Config.MIN_SIZE,
                max_size=Config.MAX_SIZE,
                kwargs={"autocommit": True, "prepare_threshold": 0}
        ) as pool:
            # 短期记忆 初始化checkpointer，并初始化表结构
            app.state.checkpointer = AsyncPostgresSaver(pool)
            await app.state.checkpointer.setup()

            # 长期记忆 初始化store，并初始化表结构
            app.state.store = AsyncPostgresStore(pool)
            await app.state.store.setup()

            # 创建ReAct Agent 并存储为单实例
            app.state.agent = await create_graph(
                checkpointer=app.state.checkpointer,
                store=app.state.store
                )
            
            yield

    except Exception as e:
        raise RuntimeError(f"服务初始化失败: {str(e)}")

    # 清理资源
    finally:
        # 关闭PostgreSQL连接池
        await pool.close()

# Initialize the FastAPI application with lifecycle management
app = FastAPI(
    title="Agent智能体后端API接口服务",
    description="基于LangGraph提供AI Agent服务",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# API接口: 运行智能体并返回大模型结果或中断数据
@app.post("/agent/invoke", response_model=AgentResponse, summary="Invoke the agent with client request data.")
async def invoke_agent(
    session_id: str = Form(...),  # 使用 Form 声明每个字段
    style: str = Form(...),
    space: str = Form(...),
    emotion: str = Form(...),
    platform: str = Form(...),
    image: Optional[UploadFile] = File(None)  # 文件字段仍使用 File
):
    """
    Endpoint to invoke the agent with client request data.
    Processes the image if provided, and invokes the agent with the input state.
    Returns user review content if the agent is interrupted.

    Args:
        request (AgentRequest): Client request data for the agent.

    Returns:
        AgentResponse: User review content if the agent is interrupted.

    Raises:
        HTTPException: If the image format is unsupported, the image file is invalid, or an internal error occurs.
    """
    image_base64 = None

    # Check if an image is uploaded
    if image is not None and image.file is not None:
        # Validate file type
        allowed_content_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
        if image.content_type not in allowed_content_types:
            raise HTTPException(status_code=400, detail=f"Unsupported image format. Supported types: {', '.join(allowed_content_types)}")

        # Read image file content
        try:
            image_data = await image.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read image file: {str(e)}")

        # Verify image integrity
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()  # Verify image is not corrupted
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Convert image data to Base64 string
        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to encode image to Base64: {str(e)}")

    # Build input state for the agent
    input_state = {
        "platform": platform,
        "image": image_base64,
        "image_type": image.content_type,
        "style": style,
        "space": space,
        "emotion": emotion
    }

    try:
        # Invoke the agent with the input state
        result = await app.state.agent.ainvoke(input_state, config={"configurable": {"thread_id": session_id}})
        review_content = result['__interrupt__'][0].value
        # review_content = '123456'
        return AgentResponse(review_content=review_content, status="pending")

    except Exception as e:
        logging.error(f"Error invoking agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# API接口: 恢复被中断的智能体运行并等待运行完成或再次中断
@app.post("/agent/resume", response_model=AgentResponse, summary="Resume the agent execution with user feedback.")
async def resume_agent(request: InterruptRequest):
    """
    Endpoint to resume the agent execution with user feedback.
    Returns user review content if the agent is interrupted again, otherwise returns the final content.

    Args:
        request (InterruptRequest): User feedback for resuming the agent.

    Returns:
        AgentResponse: User review content if the agent is interrupted again, otherwise the final content.

    Raises:
        HTTPException: If an internal error occurs.
    """
    if request.info:
        decision = "edit"
    else:
        decision = "approve"
    command_data = {
        "decision": decision,
        "info": request.info
    }
    try:
        # Resume agent execution with the provided command data
        result = await app.state.agent.ainvoke(Command(resume=command_data), config={"configurable": {"thread_id": request.session_id}})
        if result.get('__interrupt__'):
            review_content = result['__interrupt__'][0].value
            return AgentResponse(review_content=review_content, status="pending")
        final_content = result.get('text', '')
        return AgentResponse(review_content=final_content, status="completed")

    except Exception as e:
        logging.error(f"Error resuming agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# 启动服务器
if __name__ == "__main__":
    # Run the FastAPI application with specified host, port, and log level
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
