"""Multi-Agent Orchestrator——编排 Supervisor + Specialist Agents 的协作流程。

核心循环:
  1. Supervisor 分析用户请求 → TaskDecomposition
  2. 如果是 chat → 直接返回
  3. 否则按依赖顺序执行 subtasks：
     - 找到已就绪的 subtask（依赖全部完成）
     - 路由到对应 Specialist Agent
     - 收集结果
  4. 可选的 Reflection 自检 → 优化输出
  5. Agent 聚合为最终回复
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from agentkb.agents.supervisor import SupervisorAgent, TaskDecomposition, SubTask
from agentkb.agents.registry import get_agent_registry
from agentkb.agents.reflection import ReflectionModule
from agentkb.observability.tracer import get_tracer


class MultiAgentOrchestrator:
    """多 Agent 编排器——将 Supervisor 的任务分解转化为实际执行。"""

    def __init__(self, llm_client=None) -> None:
        self._supervisor = SupervisorAgent(llm_client=llm_client)
        self._llm = llm_client
        self._reflection = ReflectionModule()

    async def execute(
        self,
        query: str,
        history: list[str] | None = None,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """主入口——处理用户请求并返回结果。

        Returns:
          {
            "intent": str,
            "output": str,            # 最终回复
            "agent_results": [...],   # 各 Agent 的结果
            "trace": {...},           # 完整 trace
            "elapsed_ms": float,
          }
        """
        tracer = get_tracer()
        t0 = time.time()

        result = {
            "intent": "chat",
            "output": "",
            "agent_results": [],
            "elapsed_ms": 0,
        }




        try:
            with tracer.start_trace(session_id=session_id, query=query):
                # 1. Supervisor 分析
                with tracer.span("supervisor_analysis"):
                    decomposition = await self._supervisor.analyze(query, history)

                result["intent"] = decomposition.intent

                # 2. 闲聊直接回复
                if decomposition.direct_reply:
                    result["output"] = decomposition.direct_reply
                    result["elapsed_ms"] = (time.time() - t0) * 1000
                    return result

                # 3. 执行 subtasks（按依赖拓扑排序）
                agent_results = await self._execute_subtasks(
                    decomposition.subtasks, query, history
                )
                result["agent_results"] = [
                    r.to_dict() if hasattr(r, "to_dict") else r
                    for r in agent_results
                ]

                # 4. Reflection 自检
                with tracer.span("reflection"):
                    refined = await self._reflection.critique(
                        query=query,
                        agent_results=agent_results,
                        llm_client=self._llm,
                    )
                    if refined.get("needs_revision") and refined.get("revised_output"):
                        logger.info("Reflection 触发修订")
                        result["output"] = refined["revised_output"]
                        result["reflection"] = refined.get("critique", "")
                    else:
                        result["output"] = await self._supervisor.aggregate(
                            [r.to_dict() for r in agent_results], query
                        )

                result["elapsed_ms"] = (time.time() - t0) * 1000

        except Exception as exc:
            logger.error(f"Orchestrator 执行失败: {exc}")
            result["output"] = f"处理请求时出错：{exc}"
            result["elapsed_ms"] = (time.time() - t0) * 1000

        return result

    async def _execute_subtasks(
        self,
        subtasks: list[SubTask],
        query: str,
        history: list[str] | None,
    ) -> list:
        """按依赖顺序执行子任务列表。"""
        registry = get_agent_registry()
        tracer = get_tracer()
        completed: dict[int, Any] = {}  # subtask_id → AgentResult
        pending = list(subtasks)

        while pending:
            # 找到依赖已全部完成的 subtask
            ready = []
            still_pending = []
            for st in pending:
                if all(dep in completed for dep in st.dependencies):
                    ready.append(st)
                else:
                    still_pending.append(st)
            pending = still_pending

            if not ready:
                # 循环依赖或无 Agent，跳过剩余
                logger.warning(f"存在无法执行的子任务: {[s.id for s in pending]}")
                break

            # 并行执行所有就绪的 subtask
            tasks = []
            for st in ready:
                tasks.append(self._run_single_subtask(st, query, history, completed))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for st, res in zip(ready, results):
                if isinstance(res, Exception):
                    logger.error(f"Subtask {st.id} 异常: {res}")
                    completed[st.id] = {"error": str(res)}
                else:
                    completed[st.id] = res

        # 按原始顺序返回结果
        return [completed[st.id] for st in subtasks if st.id in completed]

    async def _run_single_subtask(
        self,
        subtask: SubTask,
        query: str,
        history: list[str] | None,
        completed: dict,
    ) -> Any:
        """执行单个子任务。"""
        registry = get_agent_registry()
        tracer = get_tracer()

        agent = registry.get(subtask.assigned_agent)
        if agent is None:
            # 尝试按 intent 查找
            candidates = registry.find_by_intent(subtask.assigned_agent)
            agent = candidates[0] if candidates else None

        if agent is None:
            logger.warning(f"Agent '{subtask.assigned_agent}' 未注册，跳过子任务 {subtask.id}")
            from agentkb.agents.base import AgentResult
            return AgentResult(
                agent_name=subtask.assigned_agent,
                success=False,
                error=f"Agent '{subtask.assigned_agent}' 未注册",
            )

        # 构建上下文（包含已完成 subtask 的结果）
        context = {
            "original_query": query,
            "history": history,
            "completed_subtasks": {
                str(st_id): {
                    "output": r.output if hasattr(r, "output") else str(r),
                }
                for st_id, r in completed.items()
            },
        }

        with tracer.span(f"agent:{agent.name}"):
            result = await agent.execute(task=subtask.description, context=context)

        return result
