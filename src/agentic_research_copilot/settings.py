from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
GITHUB_MCP_READONLY_URL = "https://api.githubcopilot.com/mcp/readonly"
GITHUB_MCP_READONLY_TOOLS = [
    "search_repositories",
    "get_file_contents",
    "search_code",
    "list_issues",
    "issue_read",
    "search_issues",
    "list_pull_requests",
    "pull_request_read",
    "get_latest_release",
]


class AppSettings(BaseModel):
    storage_path: str = Field(default=".arc/agentic_research.db")
    orchestration_runtime: Literal["langgraph"] = "langgraph"
    strict_providers: bool = False
    search_provider: Literal[
        "none",
        "duckduckgo",
        "tavily",
        "brave",
        "serpapi",
        "exa",
        "perplexity",
        "arxiv",
        "pubmed",
        "linkup",
        "openai_web",
        "anthropic_web",
    ] = "none"
    search_api_key: str = ""
    search_base_url: str = ""
    search_model: str = ""
    search_depth: str = "basic"
    search_timeout_seconds: float = 8.0
    search_max_results: int = 5
    search_include_raw_content: bool = True
    mcp_enabled: bool = False
    mcp_server_url: str = ""
    mcp_tools: list[str] = Field(default_factory=list)
    mcp_auth_required: bool = False
    mcp_auth_token: str = ""
    mcp_prompt: str = ""
    mcp_transport: Literal["streamable_http", "sse"] = "streamable_http"
    mcp_timeout_seconds: float = 20.0
    source_reader_enabled: bool = True
    source_reader_strategy: Literal["extract", "model_compress", "chunk_rerank_compress"] = "extract"
    source_reader_max_chars: int = 50000
    source_reader_excerpt_chars: int = 1600
    source_reader_chunk_context_window: int = 1
    research_max_workers: int = 4
    research_max_iterations: int = 3
    job_max_attempts: int = 2
    job_timeout_seconds: float = 120.0
    job_queue_backend: Literal["in_process", "celery"] = "in_process"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    model_provider: Literal["deterministic", "openai_compatible"] = "deterministic"
    model_base_url: str = ""
    model_api_key: str = ""
    model_chat_model: str = "gpt-4o-mini"
    model_embedding_model: str = "text-embedding-3-small"
    model_timeout_seconds: float = 30.0
    model_temperature: float = 0.2
    embedding_provider: Literal["model", "deterministic", "openai_compatible"] = "model"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimensions: int = 256
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "arc_documents"
    qdrant_location: str = ":memory:"
    qdrant_prefer_local: bool = True
    rag_max_query_rewrites: int = 2
    rag_min_evidence_per_item: int = 2
    rag_min_source_diversity: int = 2
    rag_hybrid_fusion: Literal["rrf", "dbsf"] = "rrf"
    rag_graph_enabled: bool = True
    rag_graph_max_entities_per_chunk: int = 12
    rag_graph_max_relationships_per_chunk: int = 16
    rag_graph_neighbor_limit: int = 8
    rag_graph_entity_candidate_limit: int = 8
    rag_graph_relation_candidate_limit: int = 8
    rerank_provider: Literal["rule", "dashscope", "qwen", "qwen3"] = "dashscope"
    rerank_base_url: str = DASHSCOPE_COMPATIBLE_BASE_URL
    rerank_api_key: str = ""
    rerank_model: str = "qwen3-rerank"
    rerank_timeout_seconds: float = 15.0
    rerank_candidate_limit: int = 24
    langgraph_checkpointer: Literal["sqlite", "memory"] = "sqlite"
    langgraph_checkpoint_path: str = ".arc/langgraph_checkpoints.sqlite"
    max_revisions: int = 2
    seed_reference_knowledge: bool = False


def load_settings() -> AppSettings:
    _load_dotenv_files()
    search_provider = os.getenv("ARC_SEARCH_PROVIDER", "none").lower()
    model_base_url = os.getenv("ARC_MODEL_BASE_URL", "").rstrip("/")
    return AppSettings(
        storage_path=os.getenv("ARC_STORAGE_PATH", ".arc/agentic_research.db"),
        orchestration_runtime=os.getenv("ARC_ORCHESTRATION_RUNTIME", "langgraph").lower(),
        strict_providers=_env_bool("ARC_STRICT_PROVIDERS", False)
        or _env_bool("ARC_DISABLE_PROVIDER_FALLBACKS", False),
        search_provider=search_provider,
        search_api_key=_search_api_key(search_provider),
        search_base_url=os.getenv("ARC_SEARCH_BASE_URL", "").rstrip("/"),
        search_model=os.getenv("ARC_SEARCH_MODEL", ""),
        search_depth=os.getenv("ARC_SEARCH_DEPTH", "basic"),
        search_timeout_seconds=float(os.getenv("ARC_SEARCH_TIMEOUT_SECONDS", "8")),
        search_max_results=int(os.getenv("ARC_SEARCH_MAX_RESULTS", "5")),
        search_include_raw_content=_env_bool("ARC_SEARCH_INCLUDE_RAW_CONTENT", True),
        mcp_enabled=_env_bool("ARC_MCP_ENABLED", False),
        mcp_server_url=os.getenv("ARC_MCP_SERVER_URL", "").rstrip("/"),
        mcp_tools=_env_list("ARC_MCP_TOOLS"),
        mcp_auth_required=_env_bool("ARC_MCP_AUTH_REQUIRED", False),
        mcp_auth_token=_first_env("ARC_MCP_AUTH_TOKEN", "MCP_AUTH_TOKEN"),
        mcp_prompt=os.getenv("ARC_MCP_PROMPT", ""),
        mcp_transport=os.getenv("ARC_MCP_TRANSPORT", "streamable_http").lower(),
        mcp_timeout_seconds=float(os.getenv("ARC_MCP_TIMEOUT_SECONDS", "20")),
        source_reader_enabled=_env_bool("ARC_SOURCE_READER_ENABLED", True),
        source_reader_strategy=os.getenv("ARC_SOURCE_READER_STRATEGY", "extract").lower(),
        source_reader_max_chars=int(os.getenv("ARC_SOURCE_READER_MAX_CHARS", "50000")),
        source_reader_excerpt_chars=int(os.getenv("ARC_SOURCE_READER_EXCERPT_CHARS", "1600")),
        source_reader_chunk_context_window=int(os.getenv("ARC_SOURCE_READER_CHUNK_CONTEXT_WINDOW", "1")),
        research_max_workers=int(os.getenv("ARC_RESEARCH_MAX_WORKERS", "4")),
        research_max_iterations=int(os.getenv("ARC_RESEARCH_MAX_ITERATIONS", "3")),
        job_max_attempts=int(os.getenv("ARC_JOB_MAX_ATTEMPTS", "2")),
        job_timeout_seconds=float(os.getenv("ARC_JOB_TIMEOUT_SECONDS", "120")),
        job_queue_backend=os.getenv("ARC_JOB_QUEUE_BACKEND", "in_process").lower(),
        celery_broker_url=os.getenv("ARC_CELERY_BROKER_URL", "redis://localhost:6379/0"),
        celery_result_backend=os.getenv("ARC_CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
        model_provider=os.getenv("ARC_MODEL_PROVIDER", "deterministic").lower(),
        model_base_url=model_base_url,
        model_api_key=_model_api_key(model_base_url),
        model_chat_model=os.getenv("ARC_MODEL_CHAT_MODEL", "gpt-4o-mini"),
        model_embedding_model=os.getenv("ARC_MODEL_EMBEDDING_MODEL", "text-embedding-3-small"),
        model_timeout_seconds=float(os.getenv("ARC_MODEL_TIMEOUT_SECONDS", "30")),
        model_temperature=float(os.getenv("ARC_MODEL_TEMPERATURE", "0.2")),
        embedding_provider=os.getenv("ARC_EMBEDDING_PROVIDER", "model").lower(),
        embedding_base_url=os.getenv("ARC_EMBEDDING_BASE_URL", "").rstrip("/"),
        embedding_api_key=_first_env("ARC_EMBEDDING_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"),
        embedding_model=os.getenv(
            "ARC_EMBEDDING_MODEL",
            os.getenv("ARC_MODEL_EMBEDDING_MODEL", "qwen3.7-text-embedding"),
        ),
        embedding_dimensions=int(os.getenv("ARC_EMBEDDING_DIMENSIONS", "256")),
        qdrant_url=os.getenv("ARC_QDRANT_URL", "").rstrip("/"),
        qdrant_api_key=os.getenv("ARC_QDRANT_API_KEY", ""),
        qdrant_collection=os.getenv("ARC_QDRANT_COLLECTION", "arc_documents"),
        qdrant_location=os.getenv("ARC_QDRANT_LOCATION", ":memory:"),
        qdrant_prefer_local=_env_bool("ARC_QDRANT_PREFER_LOCAL", True),
        rag_max_query_rewrites=int(os.getenv("ARC_RAG_MAX_QUERY_REWRITES", "2")),
        rag_min_evidence_per_item=int(os.getenv("ARC_RAG_MIN_EVIDENCE_PER_ITEM", "2")),
        rag_min_source_diversity=int(os.getenv("ARC_RAG_MIN_SOURCE_DIVERSITY", "2")),
        rag_hybrid_fusion=os.getenv("ARC_RAG_HYBRID_FUSION", "rrf").lower(),
        rag_graph_enabled=_env_bool("ARC_RAG_GRAPH_ENABLED", True),
        rag_graph_max_entities_per_chunk=int(os.getenv("ARC_RAG_GRAPH_MAX_ENTITIES_PER_CHUNK", "12")),
        rag_graph_max_relationships_per_chunk=int(os.getenv("ARC_RAG_GRAPH_MAX_RELATIONSHIPS_PER_CHUNK", "16")),
        rag_graph_neighbor_limit=int(os.getenv("ARC_RAG_GRAPH_NEIGHBOR_LIMIT", "8")),
        rag_graph_entity_candidate_limit=int(os.getenv("ARC_RAG_GRAPH_ENTITY_CANDIDATE_LIMIT", "8")),
        rag_graph_relation_candidate_limit=int(os.getenv("ARC_RAG_GRAPH_RELATION_CANDIDATE_LIMIT", "8")),
        rerank_provider=os.getenv("ARC_RERANK_PROVIDER", "dashscope").lower(),
        rerank_base_url=os.getenv("ARC_RERANK_BASE_URL", DASHSCOPE_COMPATIBLE_BASE_URL).rstrip("/"),
        rerank_api_key=_first_env("ARC_RERANK_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        rerank_model=os.getenv("ARC_RERANK_MODEL", "qwen3-rerank"),
        rerank_timeout_seconds=float(os.getenv("ARC_RERANK_TIMEOUT_SECONDS", "15")),
        rerank_candidate_limit=int(os.getenv("ARC_RERANK_CANDIDATE_LIMIT", "24")),
        langgraph_checkpointer=os.getenv("ARC_LANGGRAPH_CHECKPOINTER", "sqlite").lower(),
        langgraph_checkpoint_path=os.getenv("ARC_LANGGRAPH_CHECKPOINT_PATH", ".arc/langgraph_checkpoints.sqlite"),
        max_revisions=int(os.getenv("ARC_MAX_REVISIONS", "2")),
        seed_reference_knowledge=_env_bool("ARC_SEED_REFERENCE_KNOWLEDGE", False),
    )


def resolve_storage_path(path: str) -> Path:
    storage_path = Path(path)
    if not storage_path.is_absolute():
        storage_path = Path.cwd() / storage_path
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return storage_path


def _load_dotenv_files() -> None:
    if not _env_bool("ARC_LOAD_DOTENV", True):
        return
    cwd_env = Path.cwd() / ".env"
    project_env = Path(__file__).resolve().parents[2] / ".env"
    candidates = [cwd_env] if cwd_env.exists() else [project_env]
    loaded: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in loaded or not resolved.exists():
            continue
        load_dotenv(resolved, override=False, encoding="utf-8-sig")
        loaded.add(resolved)


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _env_list(name: str) -> list[str]:
    value = os.getenv(name, "")
    return [part.strip() for part in value.split(",") if part.strip()]


def _model_api_key(base_url: str) -> str:
    lower_url = base_url.lower()
    aliases: list[str] = ["ARC_MODEL_API_KEY"]
    if "deepseek" in lower_url:
        aliases.append("DEEPSEEK_API_KEY")
    if "dashscope" in lower_url or "aliyuncs" in lower_url:
        aliases.extend(["DASHSCOPE_API_KEY", "QWEN_API_KEY"])
    aliases.extend(["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"])
    return _first_env(*aliases)


def _search_api_key(provider: str) -> str:
    aliases = {
        "tavily": ["TAVILY_API_KEY"],
        "brave": ["BRAVE_API_KEY"],
        "serpapi": ["SERPAPI_API_KEY"],
        "exa": ["EXA_API_KEY"],
        "perplexity": ["PERPLEXITY_API_KEY"],
        "linkup": ["LINKUP_API_KEY"],
        "openai_web": ["OPENAI_API_KEY"],
        "anthropic_web": ["ANTHROPIC_API_KEY"],
    }
    return _first_env("ARC_SEARCH_API_KEY", *aliases.get(provider, []))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default
