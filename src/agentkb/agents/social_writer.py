"""SocialWriter——社交媒体与内容分发 Specialist Agent。

支持: 抖音脚本、小红书笔记、微信公众号文章、LinkedIn 帖子。
每类平台有独立的风格 prompt 模板。
"""

from __future__ import annotations

import time
from typing import Any

from agentkb.agents.base import SpecialistAgent, AgentResult


# 平台风格模板
PLATFORM_TEMPLATES = {
    "xiaohongshu": {
        "tone": "亲切、种草感、使用 emoji 和标签",
        "structure": "标题（吸引眼球）→ 开头（共鸣/痛点）→ 正文（干货/体验）→ 结尾（互动引导）→ 标签",
        "max_length": "800字",
        "style_notes": "大量使用 emoji，短句为主，段落间空行，用「你」拉近距离",
    },
    "douyin": {
        "tone": "快节奏、金句密集、强反差/悬念",
        "structure": "黄金3秒开头 → 冲突/问题 → 解决方案 → 结尾反转/升华 → #话题",
        "max_length": "300字（配合口语表达）",
        "style_notes": "短句、口语化、「你绝对想不到」「最后一个太绝了」风格",
    },
    "wechat": {
        "tone": "深度、有料、专业但不枯燥",
        "structure": "引言（为什么你要看）→ 正文（分段展开）→ 总结（金句）→ 引导关注",
        "max_length": "2000字",
        "style_notes": "开头要有钩子，段落分明，适当引用数据和案例，文末 CTA",
    },
    "linkedin": {
        "tone": "专业、洞察力强、启发式",
        "structure": "Hook → 个人经历/观察 → 关键洞察 → 行动建议 → 讨论引导",
        "max_length": "1300字",
        "style_notes": "避免过度销售，用真实经历引出观点，结尾留开放式问题",
    },
    "pengyouquan": {
        "tone": "真实、温暖、生活化",
        "structure": "日常场景 → 情绪感受 → 一句话升华",
        "max_length": "200字",
        "style_notes": "像对朋友说话，不要有营销感，真实最重要",
    },
}

SOCIAL_WRITER_PROMPT = """你是{platform_name}平台的资深内容创作者。请根据产品/主题创作一条{platform_name}风格的帖子。

## 平台风格指南
- 语调: {tone}
- 结构: {structure}
- 长度限制: {max_length}
- 风格要点: {style_notes}

## 创作主题
{task}

## 参考信息
{context}

## 创作要求
- 严格遵循上述平台风格
- 内容原创、有吸引力
- 适合目标平台受众
- 直接输出内容，不要额外说明"""


class SocialWriterAgent(SpecialistAgent):
    """社交媒体与内容分发专家。"""

    # 平台关键词映射
    PLATFORM_KEYWORDS = {
        "xiaohongshu": ["小红书", "redbook", "red", "种草", "测评"],
        "douyin": ["抖音", "douyin", "tiktok", "短视频", "视频脚本"],
        "wechat": ["公众号", "微信", "wechat", "订阅号", "推文"],
        "linkedin": ["linkedin", "领英", "职场分享"],
        "pengyouquan": ["朋友圈", "分享文案"],
    }

    @property
    def name(self) -> str:
        return "social_writer"

    @property
    def description(self) -> str:
        return "为不同社交平台创作优化内容：小红书笔记、抖音脚本、公众号文章、朋友圈文案"

    @property
    def intents(self) -> list[str]:
        return ["social_content"]

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        tools: list | None = None,
    ) -> AgentResult:
        t0 = time.time()
        context = context or {}
        tokens_used = 0

        try:
            # 1. 检测目标平台
            platform = self._detect_platform(task)
            template = PLATFORM_TEMPLATES[platform]

            # 2. 收集参考上下文
            ctx_text = "（无参考信息）"
            completed = context.get("completed_subtasks", {})
            for st_data in completed.values():
                output = st_data.get("output", "")
                if output:
                    ctx_text += output[:1500] + "\n"

            # 如果知识检索在先，使用其输出
            if not completed:
                try:
                    from agentkb.knowledge.retriever import get_retriever
                    retriever = get_retriever()
                    candidates = retriever.retrieve(task)
                    if candidates:
                        contents = [
                            c.get("parent_content") or c.get("content", "")
                            for c in candidates[:3]
                        ]
                        ctx_text = "\n\n".join(contents[:1500])
                except Exception:
                    pass

            # 3. 创作
            llm = self.llm
            prompt = SOCIAL_WRITER_PROMPT.format(
                platform_name=self._platform_display_name(platform),
                tone=template["tone"],
                structure=template["structure"],
                max_length=template["max_length"],
                style_notes=template["style_notes"],
                task=task,
                context=ctx_text[:2000],
            )

            resp = await llm.ainvoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            tokens_used = len(prompt) // 2 + len(content) // 2

            # 添加平台标签
            header = f"📱 **{self._platform_display_name(platform)}内容**\n\n"
            output = header + content

            return AgentResult(
                agent_name=self.name,
                success=True,
                output=output,
                data={
                    "platform": platform,
                    "word_count": len(content),
                    "template": template,
                },
                tokens_used=tokens_used,
                elapsed_ms=(time.time() - t0) * 1000,
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
                output="内容创作遇到了问题。",
                elapsed_ms=(time.time() - t0) * 1000,
            )

    def _detect_platform(self, task: str) -> str:
        """从用户输入中检测目标平台。"""
        for platform, keywords in self.PLATFORM_KEYWORDS.items():
            if any(kw in task.lower() for kw in keywords):
                return platform
        return "xiaohongshu"  # 默认小红书

    @staticmethod
    def _platform_display_name(platform: str) -> str:
        names = {
            "xiaohongshu": "小红书",
            "douyin": "抖音",
            "wechat": "微信公众号",
            "linkedin": "LinkedIn",
            "pengyouquan": "朋友圈",
        }
        return names.get(platform, platform)
