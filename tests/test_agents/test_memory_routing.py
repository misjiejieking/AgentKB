from __future__ import annotations

from agentkb.agents.supervisor import SupervisorAgent


def test_supervisor_fast_routes_explicit_memory_request():
    decomposition = SupervisorAgent._fast_route("请记住我偏好简洁回答")

    assert decomposition is not None
    assert decomposition.intent == "personal_memory"
    assert decomposition.subtasks[0].assigned_agent == "memory_agent"
