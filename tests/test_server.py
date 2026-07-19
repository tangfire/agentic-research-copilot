from fastapi.testclient import TestClient

from agentic_research_copilot.server import app


def test_root_page_includes_docs_link():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Agentic Research Copilot" in response.text
    assert "/docs" in response.text
