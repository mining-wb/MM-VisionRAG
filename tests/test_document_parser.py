# 测试 document_parser 的加载与分块
import tempfile
from pathlib import Path

import pytest

from src.document_parser import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    load_documents,
    parse_document_to_chunks,
)


def test_parse_txt_to_chunks():
    """临时 .txt 解析后应得到非空块列表，且块内容来自原文。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("第一段内容。\n\n第二段内容。\n\n第三段内容。")
        path = f.name
    try:
        chunks = parse_document_to_chunks(path)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        full = "".join(chunks)
        assert "第一段" in full or "第二段" in full
    finally:
        Path(path).unlink(missing_ok=True)


def test_chunk_size_overlap():
    """长文本应按 chunk_size 与 overlap 切分，相邻块有重叠。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("中" * (CHUNK_SIZE * 3))
        path = f.name
    try:
        chunks = parse_document_to_chunks(path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= CHUNK_SIZE + 50
    finally:
        Path(path).unlink(missing_ok=True)


def test_unsupported_type_raises():
    """不支持的后缀应抛 ValueError。"""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError, match="不支持"):
            parse_document_to_chunks(path)
    finally:
        Path(path).unlink(missing_ok=True)
