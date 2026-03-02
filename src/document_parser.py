# ====== 文档解析：.txt / PDF，滑动窗口切片 ======

CHUNK_SIZE = 500
OVERLAP = 50


def read_txt(path: str, encoding: str = "utf-8") -> str:
    """读 .txt 文件，返回全文。编码错误或空文件会抛明确异常。"""
    try:
        with open(path, "r", encoding=encoding) as f:
            text = f.read().strip()
    except UnicodeDecodeError as e:
        raise ValueError(f"文件编码异常，请用 UTF-8 保存: {path}") from e
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {path}")
    if not text:
        raise ValueError(f"文件为空: {path}")
    return text


def read_pdf(path: str) -> str:
    """用 pdfplumber 提取 PDF 全文，空页或异常时抛明确错误。"""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber") from None
    try:
        with pdfplumber.open(path) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            text = "\n\n".join(parts).strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {path}")
    except Exception as e:
        raise ValueError(f"PDF 解析失败: {path}") from e
    if not text:
        raise ValueError(f"PDF 无有效文本: {path}")
    return text


def parse_document(path: str) -> str:
    """按扩展名选 .txt 或 PDF 解析，返回全文。"""
    p = path.lower()
    if p.endswith(".txt"):
        return read_txt(path)
    if p.endswith(".pdf"):
        return read_pdf(path)
    raise ValueError(f"不支持的文件类型，仅支持 .txt / .pdf: {path}")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """按固定字数切块，块与块之间保留重叠区，避免句子在边界被截断。"""
    if overlap >= chunk_size:
        overlap = chunk_size - 1
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def parse_document_to_chunks(
    path: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """解析文档（.txt / .pdf）并做滑动窗口切片，返回块列表。"""
    text = parse_document(path)
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
