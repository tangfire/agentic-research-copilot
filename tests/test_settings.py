from agentic_research_copilot.provider_validation import validate_real_provider_config
from agentic_research_copilot.settings import DASHSCOPE_COMPATIBLE_BASE_URL, load_settings


def test_default_reranker_uses_dashscope_with_local_fallback(monkeypatch):
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
    monkeypatch.delenv("ARC_RERANK_PROVIDER", raising=False)
    monkeypatch.delenv("ARC_RERANK_BASE_URL", raising=False)
    monkeypatch.delenv("ARC_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    settings = load_settings()

    assert settings.rerank_provider == "dashscope"
    assert settings.rerank_base_url == DASHSCOPE_COMPATIBLE_BASE_URL
    assert settings.rerank_api_key == ""
    assert settings.search_include_raw_content is True
    assert settings.mcp_enabled is True
    assert settings.mcp_server_url == ""
    assert settings.mcp_tools == []
    assert settings.mcp_auth_required is False
    assert settings.mcp_auth_token == ""
    assert settings.mcp_prompt == ""
    assert settings.source_reader_enabled is True
    assert settings.source_reader_strategy == "extract"
    assert settings.source_reader_chunk_context_window == 1
    assert settings.research_max_iterations == 3
    assert settings.rag_graph_enabled is True
    assert settings.rag_graph_max_entities_per_chunk == 12
    assert settings.rag_graph_neighbor_limit == 8


def test_load_settings_reads_dotenv_and_common_key_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARC_LOAD_DOTENV", "true")
    for name in [
        "ARC_MODEL_PROVIDER",
        "ARC_MODEL_BASE_URL",
        "ARC_MODEL_API_KEY",
        "ARC_EMBEDDING_PROVIDER",
        "ARC_EMBEDDING_BASE_URL",
        "ARC_EMBEDDING_API_KEY",
        "ARC_SEARCH_PROVIDER",
        "ARC_SEARCH_API_KEY",
        "ARC_RERANK_API_KEY",
        "TAVILY_API_KEY",
        "DASHSCOPE_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ARC_MODEL_PROVIDER=openai_compatible",
                "ARC_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "ARC_EMBEDDING_PROVIDER=openai_compatible",
                "ARC_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "ARC_SEARCH_PROVIDER=tavily",
                "TAVILY_API_KEY=tavily-test-key",
                "DASHSCOPE_API_KEY=dashscope-test-key",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings()

    assert settings.model_provider == "openai_compatible"
    assert settings.model_api_key == "dashscope-test-key"
    assert settings.embedding_api_key == "dashscope-test-key"
    assert settings.search_provider == "tavily"
    assert settings.search_api_key == "tavily-test-key"
    assert settings.rerank_api_key == "dashscope-test-key"


def test_load_settings_reads_bom_prefixed_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARC_LOAD_DOTENV", "true")
    monkeypatch.delenv("ARC_STRICT_PROVIDERS", raising=False)
    monkeypatch.delenv("ARC_STORAGE_PATH", raising=False)
    (tmp_path / ".env").write_text(
        "\ufeffARC_STRICT_PROVIDERS=true\nARC_STORAGE_PATH=.arc/demo.sqlite\n",
        encoding="utf-8",
    )

    settings = load_settings()

    assert settings.strict_providers is True
    assert settings.storage_path == ".arc/demo.sqlite"


def test_strict_provider_mode_reports_missing_real_config(monkeypatch):
    monkeypatch.setenv("ARC_STRICT_PROVIDERS", "true")

    settings = load_settings()
    issues = validate_real_provider_config(settings)

    assert settings.strict_providers is True
    assert issues
    assert any(issue.field == "ARC_MODEL_PROVIDER" for issue in issues)


def test_strict_provider_mode_accepts_real_provider_shape(monkeypatch):
    monkeypatch.setenv("ARC_STRICT_PROVIDERS", "true")
    monkeypatch.setenv("ARC_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("ARC_MODEL_BASE_URL", "https://relay.example.test/v1")
    monkeypatch.setenv("ARC_MODEL_API_KEY", "chat-key")
    monkeypatch.setenv("ARC_MODEL_CHAT_MODEL", "demo-chat")
    monkeypatch.setenv("ARC_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("ARC_EMBEDDING_BASE_URL", "https://dashscope.example.test/compatible-mode/v1")
    monkeypatch.setenv("ARC_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("ARC_EMBEDDING_MODEL", "qwen3.7-text-embedding")
    monkeypatch.setenv("ARC_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("ARC_SEARCH_API_KEY", "search-key")
    monkeypatch.setenv("ARC_RERANK_PROVIDER", "dashscope")
    monkeypatch.setenv("ARC_RERANK_BASE_URL", "https://dashscope.example.test/compatible-mode/v1")
    monkeypatch.setenv("ARC_RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv("ARC_RERANK_MODEL", "qwen3-rerank")
    monkeypatch.setenv("ARC_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINTER", "sqlite")

    settings = load_settings()
    issues = validate_real_provider_config(settings)

    assert settings.strict_providers is True
    assert issues == []


def test_celery_strict_mode_requires_qdrant_server_url(monkeypatch):
    monkeypatch.setenv("ARC_STRICT_PROVIDERS", "true")
    monkeypatch.setenv("ARC_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("ARC_MODEL_BASE_URL", "https://relay.example.test/v1")
    monkeypatch.setenv("ARC_MODEL_API_KEY", "chat-key")
    monkeypatch.setenv("ARC_MODEL_CHAT_MODEL", "demo-chat")
    monkeypatch.setenv("ARC_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("ARC_EMBEDDING_BASE_URL", "https://dashscope.example.test/compatible-mode/v1")
    monkeypatch.setenv("ARC_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("ARC_EMBEDDING_MODEL", "qwen3.7-text-embedding")
    monkeypatch.setenv("ARC_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("ARC_SEARCH_API_KEY", "search-key")
    monkeypatch.setenv("ARC_RERANK_PROVIDER", "dashscope")
    monkeypatch.setenv("ARC_RERANK_BASE_URL", "https://dashscope.example.test/compatible-mode/v1")
    monkeypatch.setenv("ARC_RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv("ARC_RERANK_MODEL", "qwen3-rerank")
    monkeypatch.setenv("ARC_QDRANT_URL", "")
    monkeypatch.setenv("ARC_QDRANT_LOCATION", ".arc/qdrant-local")
    monkeypatch.setenv("ARC_JOB_QUEUE_BACKEND", "celery")
    monkeypatch.setenv("ARC_CELERY_BROKER_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ARC_CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    monkeypatch.setenv("ARC_LANGGRAPH_CHECKPOINTER", "sqlite")

    settings = load_settings()
    issues = validate_real_provider_config(settings)

    assert settings.job_queue_backend == "celery"
    assert any(issue.field == "ARC_QDRANT_URL" for issue in issues)
