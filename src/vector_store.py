# ====== 向量存储：抽象基类 + Chroma 实现 ======

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Awaitable, Callable

import chromadb
from chromadb.config import Settings


# ====== 抽象基类 ======

class BaseVectorStore(ABC):
    """向量库抽象，方便以后换成 Milvus 等，业务层不用改。"""

    @abstractmethod
    async def add(self, doc_id: str, chunks: list[str], embed_fn: Callable[[str], Awaitable[list[float]]]) -> int:
        """把文档块写入向量库，返回写入的块数。embed_fn 由外部注入，带超时在调用处控制。"""
        pass

    @abstractmethod
    async def query(
        self,
        query: str,
        top_k: int,
        embed_fn: Callable[[str], Awaitable[list[float]]],
    ) -> list[str]:
        """用 query 检索，返回最相关的 top_k 个文本块。"""
        pass


# ====== Chroma 实现 ======

class ChromaVectorStore(BaseVectorStore):
    """用本地 ChromaDB 存向量，add 时写入 Metadata（source、可选 page），query 支持按 Metadata 筛选。"""

    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "mm_vision_rag"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={"description": "rag chunks"})

    async def add(
        self,
        doc_id: str,
        chunks: list[str],
        embed_fn: Callable[[str], Awaitable[list[float]]],
        timeout: float = 30.0,
        metadatas: list[dict] | None = None,
    ) -> int:
        """对每个 chunk 调 embed_fn 拿向量（异步+超时），写入 Chroma 并带 Metadata（默认 source=doc_id）。"""
        if not chunks:
            return 0
        if metadatas is None:
            metadatas = [{"source": doc_id} for _ in chunks]
        if len(metadatas) != len(chunks):
            metadatas = [{"source": doc_id} for _ in chunks]
        tasks = [asyncio.wait_for(embed_fn(c), timeout=timeout) for c in chunks]
        vectors = await asyncio.gather(*tasks, return_exceptions=True)
        valid_vectors = []
        valid_chunks = []
        valid_ids = []
        valid_metas = []
        for i, v in enumerate(vectors):
            if isinstance(v, Exception):
                raise v
            valid_vectors.append(v)
            valid_chunks.append(chunks[i])
            valid_ids.append(f"{doc_id}_{i}")
            m = dict(metadatas[i]) if i < len(metadatas) else {"source": doc_id}
            if "source" not in m:
                m["source"] = doc_id
            valid_metas.append(m)
        self._collection.add(ids=valid_ids, documents=valid_chunks, embeddings=valid_vectors, metadatas=valid_metas)
        return len(valid_ids)

    async def query(
        self,
        query: str,
        top_k: int,
        embed_fn: Callable[[str], Awaitable[list[float]]],
        timeout: float = 10.0,
        where: dict | None = None,
    ) -> list[str]:
        """把 query 向量化后查 Chroma，可传 where 按 Metadata 筛选（如 where={\"source\": \"xxx.pdf\"}），返回文档片段列表。"""
        q_vec = await asyncio.wait_for(embed_fn(query), timeout=timeout)
        res = self._collection.query(query_embeddings=[q_vec], n_results=top_k, where=where)
        docs = res.get("documents", [[]])
        return list(docs[0]) if docs else []
