import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("ARC_LOAD_DOTENV", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


AI_PROVIDER_ENV_NAMES = [
    "ARC_LOAD_DOTENV",
    "ARC_STRICT_PROVIDERS",
    "ARC_DISABLE_PROVIDER_FALLBACKS",
    "ARC_MODEL_PROVIDER",
    "ARC_MODEL_BASE_URL",
    "ARC_MODEL_API_KEY",
    "ARC_MODEL_CHAT_MODEL",
    "ARC_EMBEDDING_PROVIDER",
    "ARC_EMBEDDING_BASE_URL",
    "ARC_EMBEDDING_API_KEY",
    "ARC_EMBEDDING_MODEL",
    "ARC_SEARCH_PROVIDER",
    "ARC_SEARCH_API_KEY",
    "ARC_RERANK_PROVIDER",
    "ARC_RERANK_BASE_URL",
    "ARC_RERANK_API_KEY",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY",
    "EXA_API_KEY",
    "PERPLEXITY_API_KEY",
    "BRAVE_API_KEY",
    "SERPAPI_API_KEY",
]


@pytest.fixture(autouse=True)
def isolate_ai_provider_env(monkeypatch):
    for name in AI_PROVIDER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARC_LOAD_DOTENV", "false")
