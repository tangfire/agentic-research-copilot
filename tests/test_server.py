from pathlib import Path
from time import sleep, time

from fastapi.testclient import TestClient

from agentic_research_copilot.dev_fixtures import FixtureResearchModelProvider
from agentic_research_copilot.pipeline import ResearchCopilot
from agentic_research_copilot.server import create_app


def wait_for_job(client: TestClient, status_url: str, timeout_seconds: float = 5.0) -> dict:
    deadline = time() + timeout_seconds
    last_status: dict | None = None
    while time() < deadline:
        response = client.get(status_url)
        assert response.status_code == 200
        last_status = response.json()
        if last_status["status"] in {"completed", "failed", "cancelled"}:
            return last_status
        sleep(0.05)
    raise AssertionError(f"Job did not finish in time: {last_status}")


def create_fixture_app():
    provider = FixtureResearchModelProvider()
    return create_app(
        copilot=ResearchCopilot(
            model_provider=provider,
            embedding_provider=provider,
        )
    )


class FailingPlannerProvider(FixtureResearchModelProvider):
    def draft_plan(self, *args, **kwargs):
        raise RuntimeError("planner unavailable token=secret-value")


def test_root_page_includes_simple_workbench_controls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "root.sqlite"))
    client = TestClient(create_fixture_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "AI Research Copilot" in response.text
    assert "/docs" in response.text
    assert "下一步" in response.text
    assert "运行摘要" in response.text
    assert "查看 Langfuse Trace" in response.text
    assert "发送" in response.text


def test_clarify_endpoint_returns_follow_up_for_vague_topic(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "clarify-server.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "clarify-server-checkpoints.sqlite"))
    client = TestClient(create_fixture_app())

    response = client.post("/v1/research/clarify", json={"topic": "RAG"})

    assert response.status_code == 200
    data = response.json()
    assert data["need_clarification"] is True
    assert data["question"]
    assert data["verification"] == ""


def test_api_can_store_documents_and_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "server.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "server-checkpoints.sqlite"))
    monkeypatch.setenv("ARC_SEED_REFERENCE_KNOWLEDGE", "true")
    client = TestClient(create_fixture_app())

    memory_response = client.get("/v1/memory")
    assert memory_response.status_code == 200
    assert memory_response.json() == []

    document_response = client.post(
        "/v1/documents",
        json={
            "title": "Resume Notes",
            "source": "notes.md",
            "snippet": "This project combines planning, retrieval, evaluation, and verification.",
        },
    )
    assert document_response.status_code == 200
    document_data = document_response.json()
    document_id = document_data["metadata"]["document_id"]

    documents_response = client.get("/v1/documents")
    assert documents_response.status_code == 200
    assert any(doc["metadata"]["document_id"] == document_id for doc in documents_response.json())

    document_search_response = client.get("/v1/documents/search?q=planning%20retrieval")
    assert document_search_response.status_code == 200
    document_search_data = document_search_response.json()
    assert document_search_data["result_count"] >= 1
    assert document_search_data["corpus_profile"]["document_count"] >= 1

    delete_document_response = client.delete(f"/v1/documents/{document_id}")
    assert delete_document_response.status_code == 200
    assert delete_document_response.json()["deleted"] is True

    missing_document_response = client.delete("/v1/documents/missing-document")
    assert missing_document_response.status_code == 404

    second_document_response = client.post(
        "/v1/documents",
        json={
            "title": "Architecture Notes",
            "source": "architecture.md",
            "snippet": "The copilot combines planning, retrieval, citations, and trace replay.",
        },
    )
    assert second_document_response.status_code == 200

    local_notes = tmp_path / "local-notes.md"
    local_notes.write_text(
        "Local reader ingestion keeps parsing separate from chunking and reranking.",
        encoding="utf-8",
    )
    ingest_response = client.post(
        "/v1/documents/ingest",
        json={
            "path": str(local_notes),
            "title": "Local Reader Notes",
            "metadata": {"kind": "reader_demo"},
        },
    )
    assert ingest_response.status_code == 200
    ingest_data = ingest_response.json()
    assert ingest_data["document_count"] == 1
    assert ingest_data["documents"][0]["metadata"]["reader"] == "plain_text"
    assert ingest_data["documents"][0]["metadata"]["kind"] == "reader_demo"

    run_response = client.post(
        "/v1/research/runs",
        json={"topic": "multi-agent research copilot", "depth": "standard"},
    )
    assert run_response.status_code == 200
    run_data = run_response.json()
    assert run_data["status"] == "completed"
    assert run_data["report"]["sections"]

    list_response = client.get("/v1/research/runs")
    assert list_response.status_code == 200
    assert list_response.json()

    get_response = client.get(f"/v1/research/runs/{run_data['run_id']}")
    assert get_response.status_code == 200

    checkpoints_response = client.get(f"/v1/research/runs/{run_data['run_id']}/checkpoints")
    assert checkpoints_response.status_code == 200
    assert checkpoints_response.json()

    evaluation_response = client.get(f"/v1/research/runs/{run_data['run_id']}/evaluation")
    assert evaluation_response.status_code == 200
    evaluation_data = evaluation_response.json()
    assert evaluation_data["citation_precision"] >= 1.0
    assert evaluation_data["plan_coverage"] >= 0.8

    trace_response = client.get(f"/v1/research/runs/{run_data['run_id']}/trace")
    assert trace_response.status_code == 200
    trace_data = trace_response.json()
    assert trace_data
    assert any(event["kind"] == "handoff" for event in trace_data)
    assert any(event["kind"] == "evaluation" for event in trace_data)

    replay_response = client.post(f"/v1/research/runs/{run_data['run_id']}/replay")
    assert replay_response.status_code == 200
    assert replay_response.json()["status"] == "completed"

    clear_documents_response = client.delete("/v1/documents")
    assert clear_documents_response.status_code == 200
    assert client.get("/v1/documents").json() == []

    clear_history_response = client.delete("/v1/research/history")
    assert clear_history_response.status_code == 200
    clear_history_data = clear_history_response.json()
    assert clear_history_data["runs_deleted"] >= 1
    assert clear_history_data["memory_removed_from_core"] is False
    assert clear_history_data["agent_memory_preserved"] is True
    assert client.get("/v1/research/runs").json() == []
    assert client.get("/v1/research/jobs").json() == []


def test_agent_session_collects_clarification_for_vague_message(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "agent-clarify.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "agent-clarify-checkpoints.sqlite"))
    client = TestClient(create_fixture_app())

    session_response = client.post("/v1/agent/sessions", json={"title": "Vague research"})
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    bundle_response = client.get(f"/v1/agent/sessions/{session_id}")
    assert bundle_response.status_code == 200
    bundle_data = bundle_response.json()
    assert bundle_data["messages"] == []
    assert bundle_data["steps"] == []
    assert bundle_data["workspace"]["workspace_id"]
    assert bundle_data["skill_catalog"]
    assert bundle_data["selected_skill"] is None
    assert {tool["name"] for tool in bundle_data["tool_registry"]} == {
        "web_search",
        "vector_retrieval",
        "mcp_tool",
    }

    skill_response = client.get("/v1/agent/skills/open_source_adoption_review")
    assert skill_response.status_code == 200
    skill_data = skill_response.json()
    assert skill_data["skill"]["skill_id"] == "open_source_adoption_review"
    assert skill_data["scripts"]

    preflight_response = client.post(
        "/v1/agent/skills/open_source_adoption_review/scripts/preflight/run",
        json={"payload": {"content": "LangGraph adoption review with Python/FastAPI team constraints."}},
    )
    assert preflight_response.status_code == 200
    preflight_data = preflight_response.json()
    assert preflight_data["skill_id"] == "open_source_adoption_review"
    assert preflight_data["script_name"] == "preflight"
    assert preflight_data["status"] == "completed"

    turn_response = client.post(f"/v1/agent/sessions/{session_id}/messages", json={"content": "RAG"})
    assert turn_response.status_code == 200
    turn_data = turn_response.json()
    assert turn_data["session"]["status"] == "collecting"
    assert turn_data["assistant_message"]["intent"] == "clarify"
    assert turn_data["assistant_message"]["content"]
    assert turn_data["plan_draft"] is None
    assert turn_data["memory_extraction_result"]["accepted"] == []
    assert [step["kind"] for step in turn_data["steps"]] == ["message", "tool_call", "planning"]
    assert turn_data["steps"][1]["title"].startswith("Skill preflight:")
    assert turn_data["steps"][1]["status"] == "completed"
    assert turn_data["steps"][2]["status"] == "skipped"


def test_agent_session_plans_memory_and_confirms_job(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "agent-flow.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "agent-flow-checkpoints.sqlite"))
    monkeypatch.setenv("ARC_SEED_REFERENCE_KNOWLEDGE", "true")
    with TestClient(create_fixture_app()) as client:
        session_response = client.post("/v1/agent/sessions", json={"title": "LangGraph adoption"})
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        content = (
            "我们团队是 5 人 Python/FastAPI，单机 Docker Compose 部署，必须可回滚。"
            "请评估 langchain-ai/langgraph 是否适合作为研究型 agent 的 workflow runtime，"
            "输出 adoption memo，并关注可观测性、checkpoint、工具循环和秋招展示价值。"
        )
        turn_response = client.post(
            f"/v1/agent/sessions/{session_id}/messages",
            json={"content": content, "max_sections": 2, "max_revisions": 0},
        )
        assert turn_response.status_code == 200
        turn_data = turn_response.json()
        assert turn_data["session"]["status"] == "awaiting_confirmation"
        assert turn_data["assistant_message"]["intent"] == "plan"
        assert turn_data["plan_draft"]["required_confirmation"] is True
        assert turn_data["session"]["workspace_id"]
        assert turn_data["session"]["selected_skill_id"] == "open_source_adoption_review"
        assert turn_data["selected_skill"]["skill_id"] == "open_source_adoption_review"
        assert turn_data["active_job"] is None
        assert any(memory["scope"] == "project" for memory in turn_data["memory_updates"])
        assert turn_data["memory_extraction_result"]["accepted"]
        assert any(step["kind"] == "planning" and step["status"] == "completed" for step in turn_data["steps"])
        assert "[project/constraint]" in turn_data["plan_draft"]["research_request"]["topic"]

        memory_response = client.get(f"/v1/agent/sessions/{session_id}/memory")
        assert memory_response.status_code == 200
        memory_items = memory_response.json()
        assert any("Python/FastAPI" in memory["content"] for memory in memory_items)

        memory_eval_response = client.get(f"/v1/agent/sessions/{session_id}/memory/evaluation")
        assert memory_eval_response.status_code == 200
        assert memory_eval_response.json()["project_constraint_count"] >= 1

        documents_response = client.get("/v1/documents")
        assert documents_response.status_code == 200
        assert any(doc["metadata"].get("kind") == "agent_memory" for doc in documents_response.json())

        confirm_response = client.post(f"/v1/agent/sessions/{session_id}/confirm-plan")
        assert confirm_response.status_code == 200
        confirm_data = confirm_response.json()
        assert confirm_data["session"]["status"] == "researching"
        assert confirm_data["active_job"]["job_id"]
        assert confirm_data["status_url"] == f"/v1/research/jobs/{confirm_data['active_job']['job_id']}/status"
        assert any(step["kind"] == "research" and step["status"] == "running" for step in confirm_data["steps"])

        status_data = wait_for_job(client, confirm_data["status_url"])
        assert status_data["status"] == "completed"
        assert status_data["run_id"]

        bundle_response = client.get(f"/v1/agent/sessions/{session_id}")
        assert bundle_response.status_code == 200
        bundle_data = bundle_response.json()
        assert bundle_data["session"]["status"] == "completed"
        assert bundle_data["session"]["active_run_id"] == status_data["run_id"]
        assert bundle_data["workspace"]["workspace_id"]
        assert bundle_data["selected_skill"]["skill_id"] == "open_source_adoption_review"
        assert bundle_data["active_run"]["report"]["sections"]
        assert bundle_data["active_run"]["evaluation"]["citation_precision"] >= 1.0
        assert bundle_data["role_assignments"]
        assert bundle_data["route_decisions"]
        assert bundle_data["evidence_ledger"]["total_evidence_count"] >= 1
        assert bundle_data["benchmark_summary"]["replay_fidelity"] == 0.0
        assert bundle_data["steps"]
        assert any(step["kind"] == "planning" for step in bundle_data["steps"])
        assert any(step["kind"] == "research" and step["status"] == "completed" for step in bundle_data["steps"])
        assert any(step["kind"] == "report" for step in bundle_data["steps"])
        assert any(step["kind"] == "evaluation" for step in bundle_data["steps"])
        assert bundle_data["tool_registry"]
        assert isinstance(bundle_data["tool_invocations"], list)
        assert bundle_data["constraint_coverage"]

        steps_response = client.get(f"/v1/agent/sessions/{session_id}/steps")
        assert steps_response.status_code == 200
        assert any(step["kind"] == "research" for step in steps_response.json())

        events_response = client.get(f"/v1/agent/sessions/{session_id}/events")
        assert events_response.status_code == 200
        events_data = events_response.json()
        assert any(event["type"] == "step" for event in events_data)
        assert any(event["kind"] == "message" for event in events_data)
        first_event_id = events_data[0]["event_id"]
        resumed_events_response = client.get(
            f"/v1/agent/sessions/{session_id}/events",
            params={"after_event_id": first_event_id, "limit": 3},
        )
        assert resumed_events_response.status_code == 200
        resumed_events = resumed_events_response.json()
        assert 0 < len(resumed_events) <= 3
        assert all(event["event_id"] != first_event_id for event in resumed_events)
        if len(events_data) > 1:
            assert resumed_events[0]["event_id"] == events_data[1]["event_id"]

        html_events_response = client.get(
            f"/v1/agent/sessions/{session_id}/events",
            headers={"accept": "text/html"},
        )
        assert html_events_response.status_code == 200
        assert "text/html" in html_events_response.headers.get("content-type", "")
        assert "会话事件" in html_events_response.text
        assert "返回研究台" in html_events_response.text
        assert "流程概览" in html_events_response.text
        assert "event-node" in html_events_response.text
        assert "data-expand-all" in html_events_response.text
        assert "format=json" in html_events_response.text

        json_events_response = client.get(
            f"/v1/agent/sessions/{session_id}/events?format=json",
            headers={"accept": "text/html"},
        )
        assert json_events_response.status_code == 200
        assert "application/json" in json_events_response.headers.get("content-type", "")
        assert isinstance(json_events_response.json(), list)

        export_response = client.get(f"/v1/agent/sessions/{session_id}/export")
        assert export_response.status_code == 200
        export_data = export_response.json()
        assert export_data["session_key"] == session_id
        assert export_data["workspace"]["workspace_id"]
        assert export_data["selected_skill"]["skill_id"] == "open_source_adoption_review"
        assert export_data["role_assignments"]
        assert export_data["route_decisions"]
        assert export_data["benchmark_summary"]["route_recall"] >= 0.0

        tool_invocations_response = client.get(f"/v1/agent/sessions/{session_id}/tool-invocations")
        assert tool_invocations_response.status_code == 200
        assert isinstance(tool_invocations_response.json(), list)

        coverage_response = client.get(f"/v1/research/runs/{status_data['run_id']}/constraint-coverage")
        assert coverage_response.status_code == 200
        assert coverage_response.json()

        harness_response = client.get(f"/v1/research/runs/{status_data['run_id']}/harness")
        assert harness_response.status_code == 200
        harness_data = harness_response.json()
        assert harness_data["role_assignments"]
        assert harness_data["route_decisions"]
        assert harness_data["evidence_ledger"]["total_evidence_count"] >= 1


def test_agent_session_surfaces_planning_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "agent-planning-failure.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "agent-planning-failure-checkpoints.sqlite"))
    provider = FailingPlannerProvider()
    client = TestClient(
        create_app(
            copilot=ResearchCopilot(
                model_provider=provider,
                embedding_provider=provider,
            )
        )
    )

    session_response = client.post("/v1/agent/sessions", json={"title": "Planner failure"})
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    turn_response = client.post(
        f"/v1/agent/sessions/{session_id}/messages",
        json={
            "content": (
                "我们团队是 5 人 Python/FastAPI，单机 Docker Compose 部署，必须可回滚。"
                "请评估 langchain-ai/langgraph 是否适合引入。"
            ),
            "max_sections": 2,
            "max_revisions": 0,
        },
    )

    assert turn_response.status_code == 200
    data = turn_response.json()
    assert data["session"]["status"] == "failed"
    assert data["assistant_message"]["intent"] == "plan"
    assert "规划阶段失败" in data["assistant_message"]["content"]
    assert "secret-value" not in data["assistant_message"]["content"]
    assert any(step["kind"] == "planning" and step["status"] == "failed" for step in data["steps"])

    bundle_response = client.get(f"/v1/agent/sessions/{session_id}")
    assert bundle_response.status_code == 200
    bundle = bundle_response.json()
    assert bundle["session"]["status"] == "failed"
    assert any(step["kind"] == "planning" and step["status"] == "failed" for step in bundle["steps"])


def test_agent_session_can_be_deleted_without_losing_project_memory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "agent-delete.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "agent-delete-checkpoints.sqlite"))
    monkeypatch.setenv("ARC_SEED_REFERENCE_KNOWLEDGE", "true")
    client = TestClient(create_fixture_app())

    session_response = client.post("/v1/agent/sessions", json={"title": "Delete me"})
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    turn_response = client.post(
        f"/v1/agent/sessions/{session_id}/messages",
        json={
            "content": (
                "我们团队是 5 人 Python/FastAPI，单机 Docker Compose 部署，必须可回滚。"
                "请评估 langchain-ai/langgraph 是否适合引入。"
            ),
            "max_sections": 2,
            "max_revisions": 0,
        },
    )
    assert turn_response.status_code == 200
    turn_data = turn_response.json()
    assert turn_data["memory_updates"]

    project_memory_before = client.get("/v1/memory?scope=project").json()
    assert any("Python/FastAPI" in item["content"] for item in project_memory_before)
    assert all(item["session_id"] is None for item in project_memory_before)
    session_memory_before = client.get("/v1/memory?scope=session").json()
    assert any(item["session_id"] == session_id for item in session_memory_before)

    delete_response = client.delete(f"/v1/agent/sessions/{session_id}")
    assert delete_response.status_code == 200
    delete_data = delete_response.json()
    assert delete_data["deleted"] is True
    assert delete_data["counts"]["sessions"] == 1
    assert delete_data["counts"]["messages"] >= 1
    assert delete_data["counts"]["steps"] >= 1
    assert delete_data["counts"]["session_memory_items"] >= 1

    assert client.get(f"/v1/agent/sessions/{session_id}").status_code == 404
    assert all(session["session_id"] != session_id for session in client.get("/v1/agent/sessions").json())
    session_memory_after = client.get("/v1/memory?scope=session").json()
    assert all(item["session_id"] != session_id for item in session_memory_after)
    project_memory_after = client.get("/v1/memory?scope=project").json()
    assert any("Python/FastAPI" in item["content"] for item in project_memory_after)
    documents_response = client.get("/v1/documents")
    assert documents_response.status_code == 200
    assert any(doc["metadata"].get("kind") == "agent_memory" for doc in documents_response.json())

    missing_delete_response = client.delete(f"/v1/agent/sessions/{session_id}")
    assert missing_delete_response.status_code == 404


def test_memory_endpoint_can_add_list_and_delete_items(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "memory-checkpoints.sqlite"))
    client = TestClient(create_fixture_app())

    create_response = client.post(
        "/v1/memory",
        json={
            "scope": "project",
            "kind": "constraint",
            "content": "团队约束：只允许本地单机部署，后端优先 FastAPI。",
        },
    )
    assert create_response.status_code == 200
    memory_id = create_response.json()["memory_id"]

    list_response = client.get("/v1/memory?scope=project")
    assert list_response.status_code == 200
    assert any(item["memory_id"] == memory_id for item in list_response.json())

    delete_response = client.delete(f"/v1/memory/{memory_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    list_after_delete = client.get("/v1/memory?scope=project")
    assert list_after_delete.status_code == 200
    assert all(item["memory_id"] != memory_id for item in list_after_delete.json())


def test_workspace_and_skill_registry_and_context_compaction(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "workspace.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "workspace-checkpoints.sqlite"))
    with TestClient(create_fixture_app()) as client:
        workspaces_response = client.get("/v1/agent/workspaces")
        assert workspaces_response.status_code == 200
        workspaces = workspaces_response.json()
        assert workspaces
        default_workspace = workspaces[0]
        assert default_workspace["workspace_id"]
        assert default_workspace["name"]

        skills_response = client.get("/v1/agent/skills")
        assert skills_response.status_code == 200
        skills = skills_response.json()
        assert {skill["skill_id"] for skill in skills} >= {
            "open_source_adoption_review",
            "architecture_tradeoff_memo",
            "demo_readiness_risk_review",
        }

        workspace_response = client.post(
            "/v1/agent/workspaces",
            json={
                "workspace_id": "team-alpha",
                "name": "Team Alpha",
                "team_context": "5 人 Python/FastAPI 团队，单机 Docker Compose 部署，必须可回滚。",
                "default_stack": ["Python", "FastAPI"],
                "deployment_constraints": ["single-node", "rollback"],
                "risk_policy": "read-only evidence only",
                "preferred_sources": ["GitHub", "official docs"],
                "disabled_tools": ["mcp_tool"],
            },
        )
        assert workspace_response.status_code == 200
        workspace_data = workspace_response.json()
        assert workspace_data["workspace_id"] == "team-alpha"
        assert workspace_data["metadata"]["user_configured"] is True

        session_response = client.post("/v1/agent/sessions", json={"title": "Workspace session", "workspace_id": "team-alpha"})
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        long_message = (
            "我们团队是 5 人 Python/FastAPI，单机 Docker Compose 部署，必须可回滚。"
            "请评估 langchain-ai/langgraph 是否适合作为研究型 agent 的 workflow runtime，并输出 adoption memo。"
            "另外要重点说明可观测性、checkpoint、工具循环、GitHub MCP、以及秋招展示价值。"
        ) * 20
        turn_response = client.post(
            f"/v1/agent/sessions/{session_id}/messages",
            json={"content": long_message, "max_sections": 2, "max_revisions": 0},
        )
        assert turn_response.status_code == 200
        turn_data = turn_response.json()
        assert turn_data["session"]["workspace_id"] == "team-alpha"
        assert turn_data["selected_skill"]["skill_id"] == "open_source_adoption_review"
        assert turn_data["session"]["context_summary"]

        bundle_response = client.get(f"/v1/agent/sessions/{session_id}")
        assert bundle_response.status_code == 200
        bundle_data = bundle_response.json()
        assert bundle_data["workspace"]["workspace_id"] == "team-alpha"
        assert any(step["title"] == "Context compacted" for step in bundle_data["steps"])

        export_response = client.get(f"/v1/agent/sessions/{session_id}/export")
        assert export_response.status_code == 200
        export_data = export_response.json()
        assert export_data["workspace"]["workspace_id"] == "team-alpha"
        assert export_data["session_key"] == session_id


def test_agent_reports_mcp_unavailable_when_token_is_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "agent-mcp.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "agent-mcp-checkpoints.sqlite"))
    monkeypatch.setenv("ARC_STRICT_PROVIDERS", "false")
    monkeypatch.setenv("ARC_MCP_ENABLED", "true")
    monkeypatch.setenv("ARC_MCP_SERVER_URL", "https://api.githubcopilot.com/mcp/readonly")
    monkeypatch.setenv("ARC_MCP_TOOLS", "search_code")
    monkeypatch.setenv("ARC_MCP_AUTH_REQUIRED", "true")
    monkeypatch.delenv("ARC_MCP_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    client = TestClient(create_fixture_app())

    session_response = client.post("/v1/agent/sessions", json={"title": "MCP status"})
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]

    bundle_response = client.get(f"/v1/agent/sessions/{session_id}")
    assert bundle_response.status_code == 200
    mcp_status = bundle_response.json()["mcp_status"]
    assert mcp_status["configured"] is True
    assert mcp_status["available"] is False
    assert mcp_status["provider"] == "github"
    assert mcp_status["display_name"] == "GitHub MCP"
    assert mcp_status["label"] == "GitHub MCP 未配置"
    assert mcp_status["auth_required"] is True
    assert mcp_status["auth_token_configured"] is False
    assert "token" in mcp_status["reason"].lower()

    tools_response = client.get("/v1/agent/tools")
    assert tools_response.status_code == 200
    mcp_tool = next(tool for tool in tools_response.json() if tool["name"] == "mcp_tool")
    assert mcp_tool["enabled"] is False
    assert mcp_tool["approval_required"] is True


def test_mcp_unavailable_creates_approval_request(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "agent-mcp-approval.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "agent-mcp-approval-checkpoints.sqlite"))
    monkeypatch.setenv("ARC_STRICT_PROVIDERS", "false")
    monkeypatch.setenv("ARC_MCP_ENABLED", "true")
    monkeypatch.setenv("ARC_MCP_SERVER_URL", "https://api.githubcopilot.com/mcp/readonly")
    monkeypatch.setenv("ARC_MCP_TOOLS", "search_code")
    monkeypatch.setenv("ARC_MCP_AUTH_REQUIRED", "true")
    monkeypatch.delenv("ARC_MCP_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    with TestClient(create_fixture_app()) as client:
        session_response = client.post("/v1/agent/sessions", json={"title": "MCP approval"})
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        message_response = client.post(
            f"/v1/agent/sessions/{session_id}/messages",
            json={
                "content": (
                    "请评估 langchain-ai/langgraph 的 GitHub 代码、issue 和 release 证据，"
                    "团队约束是 Python/FastAPI、单机 Docker Compose、必须可回滚。"
                ),
                "max_sections": 1,
                "max_revisions": 0,
            },
        )
        assert message_response.status_code == 200
        assert message_response.json()["session"]["status"] == "awaiting_confirmation"

        confirm_response = client.post(f"/v1/agent/sessions/{session_id}/confirm-plan")
        assert confirm_response.status_code == 200
        confirm_data = confirm_response.json()
        assert confirm_data["approval_requests"]
        approval = confirm_data["approval_requests"][0]
        assert approval["status"] == "pending"
        assert "token" in approval["reason"].lower()
        assert confirm_data["tool_invocations"][0]["status"] == "pending_approval"

        reject_response = client.post(f"/v1/agent/sessions/{session_id}/approvals/{approval['approval_id']}/reject")
        assert reject_response.status_code == 200
        assert reject_response.json()["status"] == "rejected"

        invocations_response = client.get(f"/v1/agent/sessions/{session_id}/tool-invocations")
        assert invocations_response.status_code == 200
        assert invocations_response.json()[0]["status"] == "skipped"


def test_job_status_result_and_runtime_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINT_PATH", str(tmp_path / "jobs-checkpoints.sqlite"))
    monkeypatch.setenv("ARC_SEED_REFERENCE_KNOWLEDGE", "true")
    with TestClient(create_fixture_app()) as client:
        job_response = client.post(
            "/v1/research/jobs",
            json={
                "topic": "agentic rag product architecture",
                "depth": "standard",
            },
        )
        assert job_response.status_code == 200
        job_data = job_response.json()
        job_id = job_data["job_id"]
        assert job_data["status"] in {"queued", "running"}
        assert job_data["status_url"] == f"/v1/research/jobs/{job_id}/status"
        assert job_data["result_url"] == f"/v1/research/jobs/{job_id}/result"

        status_data = wait_for_job(client, job_data["status_url"])
        assert status_data["status"] == "completed"
        assert status_data["job_id"] == job_id
        assert status_data["run_id"]
        assert status_data["checkpoint_count"] >= 1
        assert status_data["source_count"] >= 1

        result_response = client.get(job_data["result_url"])
        assert result_response.status_code == 200
        result_data = result_response.json()
        assert result_data["job_id"] == job_id
        assert result_data["report"]["source_index"]
        assert result_data["evaluation"]["citation_precision"] >= 1.0
        assert result_data["retrieval_routes"]

        jobs_response = client.get("/v1/research/jobs")
        assert jobs_response.status_code == 200
        assert jobs_response.json()[0]["job_id"] == job_id

        config_response = client.get("/v1/runtime/config")
        assert config_response.status_code == 200
        config_data = config_response.json()
        assert config_data["product"]["name"] == "AI Research Copilot"
        assert config_data["job_execution"]["mode"] == "background"
        assert config_data["job_execution"]["research_max_workers"] >= 1
        assert "cancelled" in config_data["job_execution"]["status_contract"]
        assert config_data["agent_session"]["confirmation_gate"]
        assert config_data["agent_session"]["step_stream"]["mode"] == "polling"
        assert config_data["agent_session"]["session_key"]
        assert config_data["agent_session"]["context_compaction"]["enabled"] is True
        assert config_data["workspace_control_plane"]["enabled"] is True
        assert "open_source_adoption_review" in config_data["skills"]["catalog"]
        assert config_data["memory"]["scopes"] == ["user", "project", "session"]
        assert "memory_items" in config_data["storage"]["persisted_objects"]
        assert "agent_run_steps" in config_data["storage"]["persisted_objects"]
        assert "agent_workspaces" in config_data["storage"]["persisted_objects"]
        assert "constraint_coverage" in config_data["storage"]["persisted_objects"]
        assert config_data["tool_policy"]["approval_model"] == "observable_hitl_v2"
        assert config_data["constraint_coverage"]["fail_threshold"] == 0.4
        assert config_data["retrieval"]["routes"] == ["external", "internal", "hybrid"]
        assert "evaluation" in config_data
        assert "source_quality_score" in config_data["evaluation"]["metrics"]
        assert "faithfulness_proxy" in config_data["evaluation"]["metrics"]
        assert config_data["retrieval"]["hybrid_pipeline"]["fusion"] in {"rrf", "dbsf"}
        assert any(agent["name"] == "research_supervisor" for agent in config_data["agents"])
        assert "research_supervisor" in config_data["orchestration"]["active_graph"]
        assert "open_deep_research_alignment" in config_data
        web_tool = next(tool for tool in config_data["tool_registry"] if tool["name"] == "web_search")
        assert web_tool["include_raw_content"] is True
        assert web_tool["reader_strategy"] == "provider_raw_content_extract"
        assert all(reference["dependency"] is False for reference in config_data["reference_designs"])
        assert config_data["provider_readiness"]["strict_providers"] is False

        provider_check_response = client.get("/v1/runtime/provider-check")
        assert provider_check_response.status_code == 200
        assert "providers" in provider_check_response.json()
