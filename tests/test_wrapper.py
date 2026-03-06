# 测试 wrapper 的 embed / generate 返回形状与类型
import pytest

from src.wrapper import MOCK_EMBED_DIM, embed, generate


@pytest.mark.asyncio
async def test_embed_returns_vector():
    """embed 应返回长度为 MOCK_EMBED_DIM 的浮点列表。"""
    out = await embed("测试文本")
    assert isinstance(out, list)
    assert len(out) == MOCK_EMBED_DIM
    assert all(isinstance(x, (int, float)) for x in out)


@pytest.mark.asyncio
async def test_generate_returns_str():
    """generate 应返回字符串。"""
    out = await generate("你好")
    assert isinstance(out, str)
    assert len(out) >= 0


@pytest.mark.asyncio
async def test_generate_with_image_param():
    """generate 接受 image_url 可选参数不报错。"""
    out = await generate("看图回答", image_url="data:image/png;base64,abc")
    assert isinstance(out, str)
