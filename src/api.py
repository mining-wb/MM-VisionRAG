# ====== FastAPI 应用入口 ======

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import database
from . import wrapper
from .config_logging import setup_logging
from .document_parser import parse_document_to_chunks
from .rag_pipeline import run_rag, TOP_K
from .schemas import ChatRequest, ChatResponse, TestEmbedRequest, UploadDocumentResponse
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# ====== 实例化 ======

app = FastAPI(
    title="MM-VisionRAG API",
    version="0.1.0",
    docs_url="/docs",
)

# 启动时配置日志
setup_logging()

_vector_store = ChromaVectorStore()
_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ====== 系统体检 ======

@app.get("/health")
def health():
    """确认后端服务存活。"""
    return {"status": "ok"}


# ====== 文档上传 ======

@app.post("/api/v1/upload", response_model=UploadDocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传 .txt / .pdf，解析后切片并写入向量库，返回 doc_id 与块数。"""
    if not file.filename:
        raise HTTPException(400, "缺少文件名")
    suf = Path(file.filename).suffix.lower()
    if suf not in (".txt", ".pdf"):
        raise HTTPException(400, "仅支持 .txt / .pdf")
    doc_id = f"{uuid.uuid4().hex}_{Path(file.filename).stem}"
    save_path = _UPLOAD_DIR / f"{doc_id}{suf}"
    try:
        content = await file.read()
        save_path.write_bytes(content)
    except Exception as e:
        logger.warning("上传写入失败: %s", e)
        raise HTTPException(500, "保存文件失败") from e
    try:
        chunks = parse_document_to_chunks(str(save_path))
    except Exception as e:
        logger.warning("文档解析失败: %s", e)
        raise HTTPException(400, f"解析失败: {e}") from e
    if not chunks:
        raise HTTPException(400, "文档无有效内容")
    try:
        n = await _vector_store.add(doc_id, chunks, wrapper.embed)
    except Exception as e:
        logger.warning("向量入库失败: %s", e)
        raise HTTPException(500, "向量入库失败") from e
    logger.info("文档已入库: doc_id=%s, chunks=%d", doc_id, n)
    return UploadDocumentResponse(status="ok", doc_id=doc_id, chunk_count=n)


# ====== 核心聊天 ======

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """核心对话接口。支持 query、image_base64、system_prompt、top_k 等，由调用方动态控制 RAG 行为。"""
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
            system_prompt=req.system_prompt,
            top_k=req.top_k if req.top_k is not None else TOP_K,
        )
    except Exception as e:
        logger.warning("RAG 调用失败: %s", e)
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
