# ====== 请求/响应数据契约 ======
# 前后端交互格式，用 Pydantic 约束

from typing import Optional
from pydantic import BaseModel, Field


# ====== 请求体 ======

class ChatRequest(BaseModel):
    """聊天接口入参。业务层通过 system_prompt、top_k 等动态控制 RAG 引擎行为，不硬编码人设。"""
    question: str = Field(..., description="用户问题（query），必填")
    image_url: Optional[str] = Field(None, description="可选，图片 URL 或 base64（image_base64），多模态用")
    doc_id: Optional[str] = Field(None, description="可选，关联的文档 ID，按文档筛选检索")
    session_id: Optional[str] = Field("default", description="会话 ID，多轮对话用")
    system_prompt: Optional[str] = Field(None, description="可选，系统提示词，由业务方注入人设/场景，不写死在底层")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="可选，检索返回条数，控制 RAG 召回数量")


class TestEmbedRequest(BaseModel):
    """切片测试：向量化接口入参，一段纯文本。"""
    text: str = Field(..., description="待向量化的文本")


# ====== 响应体 ======

class UploadDocumentResponse(BaseModel):
    """文档上传并向量化后的响应。"""
    status: str = Field("ok", description="状态")
    doc_id: str = Field(..., description="文档 ID，检索时可按 source 筛选")
    chunk_count: int = Field(..., description="写入向量库的块数")

class ChatResponse(BaseModel):
    """聊天接口出参：状态、大模型回答、检索到的文档片段。"""
    status: str = Field("ok", description="状态")
    answer: str = Field(..., description="大模型最终回复")
    retrieved_context: list[str] = Field(default_factory=list, description="RAG 检索到的文档片段")
