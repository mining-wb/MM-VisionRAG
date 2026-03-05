# ====== 视觉大模型与 Embedding API 封装 ======
# 仅将拼装好的 Prompt（或 Messages）与图片数据原样透传，不在此层硬编码任何业务角色或固定提示词

from typing import Optional

# 占位用向量维度，与常见 Embedding 一致即可
MOCK_EMBED_DIM = 384


async def embed(text: str) -> list[float]:
    """把文本向量化，返回稠密向量。未配置真实 API 时返回 Mock 向量。"""
    # TODO: 接入真实 Embedding API（如 OpenAI / 本地模型），读环境变量
    return [0.0] * MOCK_EMBED_DIM


async def generate(prompt: str, image_url: Optional[str] = None) -> str:
    """将上层拼装好的 prompt 与图片原样传给大模型 API，返回回复文本。不在此层写业务提示词。"""
    # TODO: 接入真实 VLM API（如 Qwen2.5-VL），读环境变量
    return "（当前为占位回复，请配置 wrapper 中的视觉大模型 API）"
