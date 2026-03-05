# ====== FastAPI 应用入口 ======

from fastapi import FastAPI, HTTPException

from . import database
from . import wrapper
from .rag_pipeline import run_rag
from .schemas import ChatRequest, ChatResponse, TestEmbedRequest
from .vector_store import ChromaVectorStore


# ====== 实例化 ======

app = FastAPI(
    title="MM-VisionRAG API",
    version="0.1.0",
    docs_url="/docs",
)

_vector_store = ChromaVectorStore()


# ====== 系统体检 ======

@app.get("/health")
def health():
    """确认后端服务存活。"""
    return {"status": "ok"}


# ====== 核心聊天 ======

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """接收用户问题与可选图片，走 RAG 流水线（历史 + 检索 + Prompt + VLM），返回回答与检索片段。"""
    session_id = req.session_id or "default"
    try:
        answer, retrieved_context = await run_rag(
            session_id=session_id,
            question=req.question,
            image_url=req.image_url,
            vector_store=_vector_store,
            embed_fn=wrapper.embed,
            generate_fn=wrapper.generate,
            get_history_fn=database.get_recent_messages,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    database.add_message(session_id, "user", req.question)
    database.add_message(session_id, "assistant", answer)
    return ChatResponse(
        status="ok",
        answer=answer,
        retrieved_context=retrieved_context,
    )


# ====== 切片测试 ======

@app.post("/api/test_embed")
async def test_embed(req: TestEmbedRequest):
    """测试向量化：输入文本，返回向量维度与预览。"""
    vec = await wrapper.embed(req.text)
    return {"dim": len(vec), "preview": vec[:5]}
