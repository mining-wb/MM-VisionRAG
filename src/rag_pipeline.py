# ====== RAG 流水线：取历史 → 检索 → 拼 Prompt → 调 VLM ======
# 不用 LangChain Chain，由 FastAPI 与自研逻辑掌控

import asyncio
from typing import Callable, Awaitable

# 默认最近 3 轮（6 条消息）拼进 Prompt
HISTORY_LIMIT = 6
TOP_K = 5

PROMPT_TEMPLATE = """你是一个依据参考文档回答问题的助手。请严格基于以下「参考文档」内容作答，不要编造文档中不存在的信息。

【参考文档】
{context}

【历史对话】
{history}

【当前问题】
{question}

请给出准确、简洁的回答。"""


async def _retry(coro_factory: Callable[[], Awaitable], max_retries: int = 2, base_delay: float = 1.0):
    """简单重试：失败后等 base_delay 再试，最多 max_retries 次。"""
    last = None
    for _ in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last = e
            if _ < max_retries:
                await asyncio.sleep(base_delay)
    raise last


def _build_history_text(messages: list[dict]) -> str:
    """把历史消息拼成一段文本。"""
    if not messages:
        return "（无）"
    lines = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


async def run_rag(
    session_id: str,
    question: str,
    image_url: str | None,
    vector_store,
    embed_fn: Callable[[str], Awaitable[list[float]]],
    generate_fn: Callable[[str, str | None], Awaitable[str]],
    get_history_fn: Callable[[str, int], list[dict]],
    top_k: int = TOP_K,
    history_limit: int = HISTORY_LIMIT,
) -> tuple[str, list[str]]:
    """
    执行 RAG：从 SQLite 取历史 → 向量检索 → 拼 Prompt → 调 VLM，返回 (回答, 检索片段列表)。
    """
    history = get_history_fn(session_id, limit=history_limit)
    retrieved = await vector_store.query(question, top_k=top_k, embed_fn=embed_fn)
    context = "\n\n".join(retrieved) if retrieved else "（无相关参考文档）"
    history_text = _build_history_text(history)
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        history=history_text,
        question=question,
    )

    async def _gen():
        return await generate_fn(prompt, image_url)

    answer = await _retry(_gen)
    return answer, retrieved
