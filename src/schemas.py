# ====== 请求/响应数据契约 ======
# 前后端交互格式，用 Pydantic 约束

from typing import Optional
from pydantic import BaseModel, Field


# ====== 请求体 ======

class ChatRequest(BaseModel):
    """聊天接口入参：用户问题、可选图片、可选文档 ID、会话 ID。"""
    question: str = Field(..., description="用户问题文本，必填")
    image_url: Optional[str] = Field(None, description="可选，图片 URL 或 base64，多模态用")
    doc_id: Optional[str] = Field(None, description="可选，关联的文档 ID，后续 RAG 用")
    session_id: Optional[str] = Field("default", description="会话 ID，多轮对话用，缺省为 default")


class TestEmbedRequest(BaseModel):
    """切片测试：向量化接口入参，一段纯文本。"""
    text: str = Field(..., description="待向量化的文本")


# ====== 响应体 ======

class ChatResponse(BaseModel):
    """聊天接口出参：状态、大模型回答、检索到的文档片段。"""
    status: str = Field("ok", description="状态")
    answer: str = Field(..., description="大模型最终回复")
    retrieved_context: list[str] = Field(default_factory=list, description="RAG 检索到的文档片段")
