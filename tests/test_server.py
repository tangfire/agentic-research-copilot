from pathlib import Path
from time import sleep, time

from fastapi.testclient import TestClient

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


def test_root_page_includes_docs_link(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "root.sqlite"))
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "AI Research Copilot" in response.text
    assert "/docs" in response.text
    assert "/v1/research/runs" in response.text
    assert "开始研究" in response.text


def test_api_can_store_documents_and_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "server.sqlite"))
    client = TestClient(create_app())

    memory_response = client.post(
        "/v1/memory",
        json={
            "key": "topic:positioning",
            "value": "Layered memory should distinguish session, summary, and canonical facts.",
            "tags": ["memory", "summary"],
            "layer": "summary",
            "topic": "agentic research copilot",
            "confidence": 0.8,
        },
    )
    assert memory_response.status_code == 200

    filtered_memory = client.get("/v1/memory?layer=summary")
    assert filtered_memory.status_code == 200
    assert filtered_memory.json()

    canonical_response = client.post(
        "/v1/memory",
        json={
            "key": "fact:positioning",
            "value": "The product is an AI research copilot.",
            "tags": ["canonical"],
            "layer": "canonical",
            "topic": "agentic research copilot",
            "confidence": 0.8,
        },
    )
    assert canonical_response.status_code == 200

    conflicting_response = client.post(
        "/v1/memory",
        json={
            "key": "fact:positioning",
            "value": "The product is a generic agent platform.",
            "tags": ["canonical"],
            "layer": "canonical",
            "topic": "agentic research copilot",
            "confidence": 0.8,
        },
    )
    assert conflicting_response.status_code == 200
    assert conflicting_response.json()["metadata"]["governance_status"] == "needs_review"

    governance_response = client.get("/v1/memory/governance")
    assert governance_response.status_code == 200
    governance_data = governance_response.json()
    assert governance_data["needs_review_count"] >= 1

    document_response = client.post(
        "/v1/documents",
        json={
            "title": "Resume Notes",
            "source": "notes.md",
            "snippet": "This project combines planning, retrieval, memory, and verification.",
        },
    )
    assert document_response.status_code == 200

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


def test_job_status_result_and_runtime_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARC_STORAGE_PATH", str(tmp_path / "jobs.sqlite"))
    with TestClient(create_app()) as client:
        job_response = client.post(
            "/v1/research/jobs",
            json={
                "topic": "agentic rag product architecture",
                "audience": "technical interviewer",
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
        assert config_data["retrieval"]["routes"] == ["external", "internal", "hybrid"]
        assert "evaluation" in config_data
        assert "source_quality_score" in config_data["evaluation"]["metrics"]
        assert "faithfulness_proxy" in config_data["evaluation"]["metrics"]
        assert config_data["retrieval"]["hybrid_pipeline"]["fusion"] in {"rrf", "dbsf"}
        assert all(reference["dependency"] is False for reference in config_data["reference_designs"])
        assert config_data["provider_readiness"]["strict_providers"] is False

        provider_check_response = client.get("/v1/runtime/provider-check")
        assert provider_check_response.status_code == 200
        assert "providers" in provider_check_response.json()
