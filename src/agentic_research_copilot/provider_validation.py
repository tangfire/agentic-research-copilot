from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .search import search_provider_requires_key


@dataclass(frozen=True)
class ProviderValidationIssue:
    field: str
    message: str
    severity: str = "error"


class ProviderConfigurationError(RuntimeError):
    """Raised when strict provider mode is enabled without real providers."""


def validate_real_provider_config(settings: Any) -> list[ProviderValidationIssue]:
    issues: list[ProviderValidationIssue] = []

    model_provider = getattr(settings, "model_provider", "deterministic")
    if model_provider != "openai_compatible":
        issues.append(
            ProviderValidationIssue(
                field="ARC_MODEL_PROVIDER",
                message="Strict demo mode requires ARC_MODEL_PROVIDER=openai_compatible.",
            )
        )
    if not getattr(settings, "model_base_url", ""):
        issues.append(
            ProviderValidationIssue(
                field="ARC_MODEL_BASE_URL",
                message="Set an OpenAI-compatible chat base URL for real model calls.",
            )
        )
    if not getattr(settings, "model_api_key", ""):
        issues.append(
            ProviderValidationIssue(
                field="ARC_MODEL_API_KEY",
                message="Set a chat model API key or a supported alias such as OPENAI_API_KEY, DEEPSEEK_API_KEY, DASHSCOPE_API_KEY, or QWEN_API_KEY.",
            )
        )
    if not getattr(settings, "model_chat_model", ""):
        issues.append(
            ProviderValidationIssue(
                field="ARC_MODEL_CHAT_MODEL",
                message="Set the chat model name used by the OpenAI-compatible provider.",
            )
        )

    embedding_provider = getattr(settings, "embedding_provider", "model")
    if embedding_provider == "deterministic":
        issues.append(
            ProviderValidationIssue(
                field="ARC_EMBEDDING_PROVIDER",
                message="Strict demo mode cannot use deterministic embeddings.",
            )
        )
    elif embedding_provider == "openai_compatible":
        if not (getattr(settings, "embedding_base_url", "") or getattr(settings, "model_base_url", "")):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_EMBEDDING_BASE_URL",
                    message="Set an OpenAI-compatible embedding base URL or reuse ARC_MODEL_BASE_URL.",
                )
            )
        if not (getattr(settings, "embedding_api_key", "") or getattr(settings, "model_api_key", "")):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_EMBEDDING_API_KEY",
                    message="Set an embedding API key or a supported alias such as DASHSCOPE_API_KEY or QWEN_API_KEY.",
                )
            )
        if not getattr(settings, "embedding_model", ""):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_EMBEDDING_MODEL",
                    message="Set the embedding model name.",
                )
            )
    elif embedding_provider == "model" and model_provider != "openai_compatible":
        issues.append(
            ProviderValidationIssue(
                field="ARC_EMBEDDING_PROVIDER",
                message="ARC_EMBEDDING_PROVIDER=model is only real when the model provider is OpenAI-compatible.",
            )
        )

    search_provider = getattr(settings, "search_provider", "none")
    if search_provider in {"none", "duckduckgo"}:
        issues.append(
            ProviderValidationIssue(
                field="ARC_SEARCH_PROVIDER",
                message="Strict demo mode requires a configured external search provider such as tavily, brave, serpapi, exa, perplexity, linkup, openai_web, or anthropic_web.",
            )
        )
    elif search_provider_requires_key(search_provider) and not getattr(settings, "search_api_key", ""):
        issues.append(
            ProviderValidationIssue(
                field="ARC_SEARCH_API_KEY",
                message=f"Set an API key for the configured search provider: {search_provider}.",
            )
        )

    rerank_provider = getattr(settings, "rerank_provider", "rule")
    if rerank_provider == "rule":
        issues.append(
            ProviderValidationIssue(
                field="ARC_RERANK_PROVIDER",
                message="Strict demo mode requires a real reranker such as dashscope/qwen/qwen3.",
            )
        )
    else:
        if not getattr(settings, "rerank_base_url", ""):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_RERANK_BASE_URL",
                    message="Set the reranker base URL.",
                )
            )
        if not getattr(settings, "rerank_api_key", ""):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_RERANK_API_KEY",
                    message="Set a reranker API key or a supported alias such as DASHSCOPE_API_KEY or QWEN_API_KEY.",
                )
            )
        if not getattr(settings, "rerank_model", ""):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_RERANK_MODEL",
                    message="Set the reranker model name.",
                )
            )

    if not getattr(settings, "qdrant_url", "") and getattr(settings, "qdrant_location", ":memory:") == ":memory:":
        issues.append(
            ProviderValidationIssue(
                field="ARC_QDRANT_URL",
                message="Strict demo mode requires a real Qdrant service URL or a persistent Qdrant location, not the in-memory fallback.",
            )
        )

    if getattr(settings, "job_queue_backend", "in_process") == "celery":
        if not getattr(settings, "celery_broker_url", ""):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_CELERY_BROKER_URL",
                    message="Celery job queue mode requires a Redis broker URL.",
                )
            )
        if not getattr(settings, "celery_result_backend", ""):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_CELERY_RESULT_BACKEND",
                    message="Celery job queue mode requires a Redis result backend URL.",
                )
            )
        if not getattr(settings, "qdrant_url", ""):
            issues.append(
                ProviderValidationIssue(
                    field="ARC_QDRANT_URL",
                    message="Celery API/worker process separation requires ARC_QDRANT_URL; embedded Qdrant paths are single-process only.",
                )
            )

    if getattr(settings, "langgraph_checkpointer", "memory") != "sqlite":
        issues.append(
            ProviderValidationIssue(
                field="ARC_LANGGRAPH_CHECKPOINTER",
                message="Use ARC_LANGGRAPH_CHECKPOINTER=sqlite for single-node durable graph checkpoints in strict demo mode.",
            )
        )

    return issues


def require_real_provider_config(settings: Any) -> None:
    issues = validate_real_provider_config(settings)
    if not issues:
        return
    formatted = "; ".join(f"{issue.field}: {issue.message}" for issue in issues)
    raise ProviderConfigurationError(f"Strict provider mode is not ready: {formatted}")


def provider_runtime_report(settings: Any) -> dict[str, object]:
    issues = validate_real_provider_config(settings)
    return {
        "strict_providers": bool(getattr(settings, "strict_providers", False)),
        "ready": not issues,
        "issues": [asdict(issue) for issue in issues],
        "providers": {
            "model": {
                "provider": getattr(settings, "model_provider", ""),
                "base_url_configured": bool(getattr(settings, "model_base_url", "")),
                "api_key_configured": bool(getattr(settings, "model_api_key", "")),
                "chat_model": getattr(settings, "model_chat_model", ""),
            },
            "embedding": {
                "provider": getattr(settings, "embedding_provider", ""),
                "base_url_configured": bool(
                    getattr(settings, "embedding_base_url", "") or getattr(settings, "model_base_url", "")
                ),
                "api_key_configured": bool(
                    getattr(settings, "embedding_api_key", "") or getattr(settings, "model_api_key", "")
                ),
                "model": getattr(settings, "embedding_model", ""),
            },
            "search": {
                "provider": getattr(settings, "search_provider", ""),
                "api_key_configured": bool(getattr(settings, "search_api_key", "")),
                "base_url_configured": bool(getattr(settings, "search_base_url", "")),
                "model": getattr(settings, "search_model", ""),
            },
            "rerank": {
                "provider": getattr(settings, "rerank_provider", ""),
                "base_url_configured": bool(getattr(settings, "rerank_base_url", "")),
                "api_key_configured": bool(getattr(settings, "rerank_api_key", "")),
                "model": getattr(settings, "rerank_model", ""),
            },
            "qdrant": {
                "url_configured": bool(getattr(settings, "qdrant_url", "")),
                "location": getattr(settings, "qdrant_location", ""),
                "prefer_local": bool(getattr(settings, "qdrant_prefer_local", False)),
            },
            "langgraph": {
                "checkpointer": getattr(settings, "langgraph_checkpointer", ""),
                "checkpoint_path": getattr(settings, "langgraph_checkpoint_path", ""),
            },
            "queue": {
                "backend": getattr(settings, "job_queue_backend", ""),
                "broker_configured": bool(getattr(settings, "celery_broker_url", "")),
                "result_backend_configured": bool(getattr(settings, "celery_result_backend", "")),
            },
        },
    }
