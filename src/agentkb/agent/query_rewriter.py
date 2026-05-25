"""查询重写——指代消解 + BM25 关键词提取，用于多轮对话场景提升召回。"""

from __future__ import annotations

from loguru import logger

REWRITE_PROMPT = """基于对话历史，将用户问题改写为自包含的检索查询。

## 规则
- 将代词（"它"、"那个"、"这个"、"他"）替换为具体实体
- 补充省略的主语和上下文
- 提取 2～4 个 BM25 检索关键词
- 输出 JSON 格式，不要输出其他内容

## 对话历史
{history}

## 用户最新问题
{query}

## 输出格式
{{"rewritten": "改写后的完整查询", "keywords": "关键词1 关键词2 关键词3"}}"""


async def rewrite_query(
    query: str,
    history: list[str] | None = None,
    llm_client=None,
) -> dict[str, str]:
    """改写用户查询——指代消解 + 关键词提取。

    Args:
        query: 用户当前问题
        history: 最近 3 轮对话消息（用户 + AI 交替）
        llm_client: LLM client（.ainvoke 接口）

    Returns:
        {"rewritten": "改写后查询", "keywords": "关键词1 关键词2"}
    """
    # 单轮对话不需要改写
    if not history or len(history) < 2:
        return {"rewritten": query, "keywords": query}

    # 截取最近 3 轮（6 条消息）
    recent = history[-6:]
    history_text = "\n".join(recent)

    # 检查是否有指代词需要消解
    pronouns = ["它", "那个", "这个", "他", "她", "它们", "这些", "那些", "其"]
    needs_rewrite = any(p in query for p in pronouns)
    if not needs_rewrite:
        return {"rewritten": query, "keywords": query}

    if llm_client is None:
        return {"rewritten": query, "keywords": query}

    prompt = REWRITE_PROMPT.format(history=history_text, query=query)

    try:
        import json
        response = await llm_client.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        # 提取 JSON（容错：去掉可能的 markdown 代码块包裹）
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        result = json.loads(content)
        rewritten = result.get("rewritten", query)
        keywords = result.get("keywords", query)
        logger.debug(f"查询改写: '{query}' → '{rewritten}' | 关键词: '{keywords}'")
        return {"rewritten": rewritten, "keywords": keywords}
    except Exception as e:
        logger.warning(f"查询改写失败，使用原查询: {e}")
        return {"rewritten": query, "keywords": query}
