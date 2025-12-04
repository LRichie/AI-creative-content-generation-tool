import os


class Config:
    """统一的配置类，集中管理所有常量"""

    # PostgreSQL数据库配置参数
    DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@120.26.185.83:5432/postgres?sslmode=disable")
    MIN_SIZE = 5
    MAX_SIZE = 10

    # openai:调用gpt模型,qwen:调用阿里通义千问大模型,oneapi:调用oneapi方案支持的模型,ollama:调用本地开源大模型
    LLM_TYPE = "openai"

    # 工作流配置
    NODE_ATTEMPTS = 5

    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8001