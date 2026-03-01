# ====== FastAPI 应用入口 ======

from fastapi import FastAPI

from .schemas import ChatRequest, ChatResponse, TestEmbedRequest


# ====== 实例化 ======

app = FastAPI(
    title="MM-VisionRAG API",
    version="0.1.0",
    docs_url="/docs",
)


# ====== 系统体检 ======

@app.get("/health")
def health():
    """确认后端服务存活。"""
    return {"status": "ok"}


# ====== 核心聊天（MVP 先返回假数据） ======

@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """接收用户问题与可选图片，返回大模型回答与检索片段。当前硬编码假数据，便于前后端先跑通。"""
    return ChatResponse(
        status="ok",
        answer="这是一个测试回复",
        retrieved_context=["假检索片段一", "假检索片段二"],
    )


# ====== 切片测试：向量化占位 ======

@app.post("/api/test_embed")
def test_embed(req: TestEmbedRequest):
    """接收一段纯文本并原样返回，后续接向量化逻辑时再改。"""
    return {"text": req.text}
