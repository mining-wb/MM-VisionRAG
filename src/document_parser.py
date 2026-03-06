# ====== 文档解析：LangChain Loaders + RecursiveCharacterTextSplitter ======
# 仅用 LangChain 做加载与分块，不做 Chain/Agent 等调度

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _get_loader(path: str):
    """按扩展名返回对应 Loader 实例，不执行 load。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return TextLoader(path, encoding="utf-8")
    if suffix == ".pdf":
        return PyMuPDFLoader(path)
    raise ValueError(f"不支持的文件类型，仅支持 .txt / .pdf: {path}")


def _get_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """滑动窗口式分块，保证语义完整、避免手写正则。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )


def load_documents(path: str):
    """用 LangChain Loader 加载文档，返回 Document 列表（含 metadata，如 page）。"""
    loader = _get_loader(path)
    docs = loader.load()
    if not docs:
        raise ValueError(f"文档无有效内容: {path}")
    return docs


def parse_document_to_chunks(
    path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """解析 .txt / .pdf 并用 RecursiveCharacterTextSplitter 分块，返回文本块列表。"""
    docs = load_documents(path)
    splitter = _get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split = splitter.split_documents(docs)
    out = [d.page_content for d in split if d.page_content.strip()]
    logger.info("解析文档 %s -> %d 块", path, len(out))
    return out
