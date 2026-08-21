from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


_GITHUB_URL_RE = re.compile(
    r"github\.com[:/](?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)",
    flags=re.IGNORECASE,
)
_SLUG_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?P<owner>[A-Za-z0-9][A-Za-z0-9_.-]{0,80})/"
    r"(?P<repo>[A-Za-z0-9_.-]{1,120})(?![A-Za-z0-9_.-])"
)
_GENERIC_REPO_PAIRS = {
    ("api", "runtime"),
    ("api", "v1"),
    ("docs", "api"),
    ("docs", "runtime"),
    ("issue", "release"),
    ("issue", "releases"),
    ("issues", "release"),
    ("issues", "releases"),
    ("pr", "release"),
    ("pr", "releases"),
    ("prs", "release"),
    ("prs", "releases"),
    ("pull", "request"),
    ("pulls", "requests"),
    ("owner", "name"),
    ("repo", "url"),
    ("readme", "license"),
}


def parse_github_repository_hint(*values: Any) -> dict[str, str] | None:
    """Extract a canonical GitHub owner/repo hint from structured or textual context."""
    structured = _parse_structured_hint(values)
    if structured is not None:
        return structured

    texts = [text for text in _flatten_text(values) if text]
    if not texts:
        return None
    combined = "\n".join(texts)

    url_hint = _first_valid_match(_GITHUB_URL_RE.finditer(combined))
    if url_hint is not None:
        return url_hint

    # Prefer slugs that appear near GitHub/repository wording, but keep bare slugs
    # available for explicit user topics like "langchain-ai/langgraph".
    candidate_texts = [
        text
        for text in texts
        if _has_repository_signal(text) or _looks_like_slug_only(text)
    ] or texts
    for text in candidate_texts:
        hint = _first_valid_match(_SLUG_RE.finditer(text))
        if hint is not None:
            return hint
    return None


def canonical_repository_slug(hint: dict[str, Any] | None) -> str | None:
    if not hint:
        return None
    owner = str(hint.get("owner") or "").strip()
    repo = str(hint.get("repo") or "").strip()
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def _parse_structured_hint(values: Iterable[Any]) -> dict[str, str] | None:
    for value in values:
        if not isinstance(value, dict):
            continue
        direct = _clean_repo_hint(value.get("owner"), value.get("repo"))
        if direct is not None:
            return direct
        for key in (
            "github_repository",
            "target_repository",
            "recognized_repo",
            "target_repo",
            "github_repository_slug",
            "repository",
            "repo",
            "repo_slug",
            "full_name",
            "html_url",
            "url",
        ):
            nested = value.get(key)
            if isinstance(nested, dict):
                nested_hint = _parse_structured_hint([nested])
                if nested_hint is not None:
                    return nested_hint
            if isinstance(nested, str):
                nested_hint = parse_github_repository_hint(nested)
                if nested_hint is not None:
                    return nested_hint
    return None


def _flatten_text(values: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for value in values:
        if isinstance(value, str):
            cleaned = " ".join(value.split())
            if cleaned:
                texts.append(cleaned)
        elif isinstance(value, dict):
            for nested in value.values():
                texts.extend(_flatten_text([nested]))
        elif isinstance(value, (list, tuple, set)):
            texts.extend(_flatten_text(value))
    return texts


def _has_repository_signal(text: str) -> bool:
    lower = text.lower()
    return any(
        signal in lower
        for signal in (
            "github",
            "repo",
            "repository",
            "仓库",
            "代码库",
            "开源项目",
        )
    )


def _looks_like_slug_only(text: str) -> bool:
    cleaned = text.strip().strip("`'\".,;:()[]{}<>")
    return bool(_SLUG_RE.fullmatch(cleaned))


def _first_valid_match(matches: Iterable[re.Match[str]]) -> dict[str, str] | None:
    for match in matches:
        hint = _clean_repo_hint(match.group("owner"), match.group("repo"))
        if hint is not None:
            return hint
    return None


def _clean_repo_hint(owner: Any, repo: Any) -> dict[str, str] | None:
    if owner is None or repo is None:
        return None
    cleaned_owner = str(owner).strip().strip("`'\".,;:()[]{}<>/")
    cleaned_repo = str(repo).strip().strip("`'\".,;:()[]{}<>/").removesuffix(".git")
    if not cleaned_owner or not cleaned_repo:
        return None
    pair = (cleaned_owner.lower(), cleaned_repo.lower())
    if pair in _GENERIC_REPO_PAIRS:
        return None
    if "." in cleaned_owner or cleaned_owner in {"http", "https", "www"}:
        return None
    return {"owner": cleaned_owner, "repo": cleaned_repo}
