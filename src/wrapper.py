# ====== 视觉大模型与 Embedding API 封装 ======
# 仅将拼装好的 Prompt（或 Messages）与图片数据原样透传，不在此层硬编码任何业务角色或固定提示词
# 密钥从项目根目录 .env 加载，.env 已加入 .gitignore，切勿提交

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

# 从项目根目录加载 .env（uvicorn 默认以项目根为 cwd）
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# 占位用向量维度，与常见 Embedding 一致即可
MOCK_EMBED_DIM = 384

# 大模型 API：从 .env 读取 API_KEY / API_URL / MODEL_NAME（兼容硅基流动等 OpenAI 兼容端点）
_API_KEY = os.getenv("API_KEY", "")
_API_URL = os.getenv("API_URL", "")
_MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3.2")


def _get_llm_client() -> AsyncOpenAI | None:
    if not _API_KEY.strip() or not _API_URL.strip():
        return None
    # OpenAI 客户端需要 base_url 不含 /chat/completions，仅到 /v1
    base_url = _API_URL.rstrip("/").replace("/chat/completions", "").rstrip("/")
    return AsyncOpenAI(api_key=_API_KEY, base_url=base_url or None)


async def embed(text: str) -> list[float]:
    """把文本向量化，返回稠密向量。未配置真实 API 时返回 Mock 向量。"""
    # TODO: 接入 Embedding API（如 OpenAI text-embedding / 本地 BGE 等），与 DeepSeek 解耦
    return [0.0] * MOCK_EMBED_DIM


async def generate(prompt: str, image_url: Optional[str] = None) -> str:
    """将上层拼装好的 prompt 与图片原样传给大模型 API，返回回复文本。"""
    client = _get_llm_client()
    if not client:
        return "（未配置 API_KEY 或 API_URL，请在 .env 中设置）"

    # 多模态：有图片时按 OpenAI 多模态 message 格式组 content
    if image_url:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        content = prompt

    try:
        resp = await client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
        )
        if not resp.choices:
            return "（模型未返回内容）"
        msg = resp.choices[0].message
        return (msg.content or "").strip()
    except Exception as e:
        return f"（调用大模型 API 失败: {e}）"
