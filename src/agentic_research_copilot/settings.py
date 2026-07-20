from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class AppSettings(BaseModel):
    storage_path: str = Field(default=".arc/agentic_research.db")
    orchestration_runtime: Literal["langgraph", "custom"] = "langgraph"
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
    research_max_workers: int = 4
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
    rerank_provider: Literal["rule", "dashscope", "qwen", "qwen3"] = "dashscope"
    rerank_base_url: str = DASHSCOPE_COMPATIBLE_BASE_URL
    rerank_api_key: str = ""
    rerank_model: str = "qwen3-rerank"
    rerank_timeout_seconds: float = 15.0
    rerank_candidate_limit: int = 24
    langgraph_checkpointer: Literal["sqlite", "memory"] = "sqlite"
    langgraph_checkpoint_path: str = ".arc/langgraph_checkpoints.sqlite"
    max_revisions: int = 2


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
        research_max_workers=int(os.getenv("ARC_RESEARCH_MAX_WORKERS", "4")),
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
        rerank_provider=os.getenv("ARC_RERANK_PROVIDER", "dashscope").lower(),
        rerank_base_url=os.getenv("ARC_RERANK_BASE_URL", DASHSCOPE_COMPATIBLE_BASE_URL).rstrip("/"),
        rerank_api_key=_first_env("ARC_RERANK_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        rerank_model=os.getenv("ARC_RERANK_MODEL", "qwen3-rerank"),
        rerank_timeout_seconds=float(os.getenv("ARC_RERANK_TIMEOUT_SECONDS", "15")),
        rerank_candidate_limit=int(os.getenv("ARC_RERANK_CANDIDATE_LIMIT", "24")),
        langgraph_checkpointer=os.getenv("ARC_LANGGRAPH_CHECKPOINTER", "sqlite").lower(),
        langgraph_checkpoint_path=os.getenv("ARC_LANGGRAPH_CHECKPOINT_PATH", ".arc/langgraph_checkpoints.sqlite"),
        max_revisions=int(os.getenv("ARC_MAX_REVISIONS", "2")),
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
        load_dotenv(resolved, override=False)
        loaded.add(resolved)


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


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
