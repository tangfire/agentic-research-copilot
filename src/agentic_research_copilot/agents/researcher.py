from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..provider_base import ResearchModelProvider
from ..schemas import EvidenceItem, MCPToolDescriptor, PlanItem
from ..source_reader import SourceReader, SourceReaderStrategy
from ..github_repository import parse_github_repository_hint


@dataclass
class ResearchCollection:
    evidence: list[EvidenceItem] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    completed_reason: str = "not_started"
    follow_up_queries: list[str] = field(default_factory=list)


class ResearchAgent:
    def __init__(
        self,
        search_tool: Callable[[str], list[dict[str, object]]] | None = None,
        *,
        model_provider: ResearchModelProvider | None = None,
        embedding_provider: ResearchModelProvider | None = None,
        mcp_tool: Callable[..., list[dict[str, object]]] | None = None,
        mcp_tool_catalog: list[MCPToolDescriptor] | tuple[MCPToolDescriptor, ...] = (),
        source_reader_enabled: bool = True,
        source_reader_strategy: SourceReaderStrategy = "extract",
        raw_content_max_chars: int = 50000,
        excerpt_max_chars: int = 1600,
        chunk_context_window: int = 1,
        max_iterations: int = 3,
    ) -> None:
        self.search_tool = search_tool
        self.mcp_tool = mcp_tool
        self.mcp_tool_catalog = list(mcp_tool_catalog)
        self.model_provider = model_provider
        self.source_reader_enabled = source_reader_enabled
        self.max_iterations = max(1, max_iterations)
        self.source_reader = SourceReader(
            strategy=source_reader_strategy,
            model_provider=model_provider,
            embedding_provider=embedding_provider,
            raw_content_max_chars=raw_content_max_chars,
            excerpt_max_chars=excerpt_max_chars,
            chunk_context_window=chunk_context_window,
        )

    def collect(self, item: PlanItem, query: str | None = None) -> list[EvidenceItem]:
        resolved_query = query or item.search_query or item.question
        if self.search_tool is None:
            return []

        results = self.search_tool(resolved_query)
        evidence: list[EvidenceItem] = []
        for result in results:
            metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
            raw_content = str(result.get("raw_content") or "")
            content = result.get("content") if result.get("content") is None else str(result.get("content"))
            snippet = result.get("snippet") if result.get("snippet") is None else str(result.get("snippet"))
            if self.source_reader_enabled and raw_content:
                read_result = self.source_reader.read(
                    query=resolved_query,
                    title=str(result.get("title", item.question)),
                    url=result.get("url") if isinstance(result.get("url"), str) else None,
                    raw_content=raw_content,
                    fallback=" ".join(part for part in [snippet or "", content or ""] if part),
                )
                if read_result is not None:
                    content = read_result.content
                    snippet = read_result.snippet
                    metadata = {
                        **metadata,
                        **read_result.metadata,
                    }

            evidence.append(
                EvidenceItem(
                    title=str(result.get("title", item.question)),
                    source=str(result.get("source", "web")),
                    kind=str(result.get("kind", "web")),
                    url=result.get("url"),
                    snippet=snippet,
                    content=content,
                    score=float(result.get("score", 0.7)),
                    metadata={
                        **metadata,
                        "plan_item_id": item.id,
                        "source_channel": "external",
                        "search_query": resolved_query,
                    },
                )
            )
        return evidence

    def collect_mcp(
        self,
        item: PlanItem,
        query: str | None = None,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        repository_hint: dict[str, str] | None = None,
    ) -> list[EvidenceItem]:
        resolved_query = query or item.search_query or item.question
        if self.mcp_tool is None:
            return []

        normalized_tool_args = self._normalize_mcp_call_args(
            item,
            tool_name=tool_name,
            tool_args=tool_args,
            repository_hint=repository_hint,
        )
        try:
            if normalized_tool_args:
                results = self.mcp_tool(resolved_query, tool_name, normalized_tool_args)
            else:
                results = self.mcp_tool(resolved_query, tool_name)
        except Exception:
            return []
        evidence: list[EvidenceItem] = []
        for result in results:
            metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
            content = result.get("content") if result.get("content") is None else str(result.get("content"))
            snippet = result.get("snippet") if result.get("snippet") is None else str(result.get("snippet"))
            evidence_metadata = {
                **metadata,
                "plan_item_id": item.id,
                "source_channel": "mcp",
                "search_query": resolved_query,
            }
            if normalized_tool_args:
                evidence_metadata["mcp_tool_args"] = normalized_tool_args
            evidence.append(
                EvidenceItem(
                    title=str(result.get("title", item.question)),
                    source=str(result.get("source", "mcp")),
                    kind=str(result.get("kind", "mcp")),
                    url=result.get("url") if isinstance(result.get("url"), str) else None,
                    snippet=snippet,
                    content=content,
                    score=float(result.get("score", 0.72)),
                    metadata=evidence_metadata,
                )
            )
        return evidence

    def _normalize_mcp_call_args(
        self,
        item: PlanItem,
        *,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
        repository_hint: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        normalized = dict(tool_args or {})
        repo_hint = repository_hint or self._github_repository_hint(item)
        if tool_name in {
            "get_file_contents",
            "search_code",
            "list_issues",
            "issue_read",
            "search_issues",
            "list_pull_requests",
            "pull_request_read",
            "get_latest_release",
        } and repo_hint is not None:
            normalized["owner"] = repo_hint["owner"]
            normalized["repo"] = repo_hint["repo"]
        if tool_name == "get_file_contents":
            normalized.setdefault("path", "README.md")
        return self._clean_mcp_args(normalized)

    def _github_repository_hint(self, item: PlanItem) -> dict[str, str] | None:
        return parse_github_repository_hint(item.question, item.purpose, item.search_query or "")

    def _clean_mcp_args(self, args: dict[str, Any]) -> dict[str, Any] | None:
        cleaned: dict[str, Any] = {}
        for key, value in args.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            cleaned_value = self._clean_mcp_arg_value(value)
            if cleaned_value in (None, "", [], {}):
                continue
            cleaned[key_text] = cleaned_value
        return cleaned or None

    def _clean_mcp_arg_value(self, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        if isinstance(value, dict):
            return self._clean_mcp_args(value)
        if isinstance(value, list):
            cleaned = [self._clean_mcp_arg_value(item) for item in value]
            cleaned = [item for item in cleaned if item is not None]
            return cleaned or None
        if isinstance(value, (bool, int, float)) or value is None:
            return value
        cleaned = str(value).strip()
        return cleaned or None

    def collect_iterative(
        self,
        item: PlanItem,
        queries: list[str] | tuple[str, ...],
        *,
        min_evidence: int = 1,
        min_sources: int = 1,
        max_iterations: int | None = None,
        repository_hint: dict[str, str] | None = None,
        required_tools: list[str] | tuple[str, ...] | None = None,
    ) -> ResearchCollection:
        """Run a bounded ODR-style think/search/MCP/complete loop for one research unit."""
        query_queue = [query.strip() for query in queries if query and query.strip()]
        if not query_queue:
            query_queue = [item.search_query or item.question]

        evidence: list[EvidenceItem] = []
        iterations: list[dict[str, Any]] = []
        follow_up_queries: list[str] = []
        previous_queries: list[str] = []
        iteration_budget = max(1, max_iterations or self.max_iterations)
        completed_reason = "query_budget_exhausted"
        required_tool_set = {
            tool for tool in (required_tools or []) if tool in {"web_search", "mcp_tool"}
        }
        available_tools: list[str] = []
        if self.search_tool is not None and (not required_tool_set or "web_search" in required_tool_set):
            available_tools.append("web_search")
        if self.mcp_tool is not None and (not required_tool_set or "mcp_tool" in required_tool_set):
            available_tools.append("mcp_tool")
        active_required_tools = tuple(tool for tool in required_tool_set if tool in available_tools)

        for iteration_index in range(iteration_budget):
            gaps_before = self._sufficiency_gaps(
                evidence,
                min_evidence=min_evidence,
                min_sources=min_sources,
                required_tools=active_required_tools,
            )
            if self.model_provider is None:
                raise RuntimeError("ResearchAgent.collect_iterative requires an explicit model provider.")
            decision, usage = self.model_provider.decide_researcher_action(
                item=item,
                available_tools=available_tools,
                previous_queries=previous_queries,
                evidence=evidence,
                gaps=gaps_before,
                iteration=iteration_index + 1,
                max_iterations=iteration_budget,
                mcp_tools=self.mcp_tool_catalog,
            )
            action = self._resolve_tool_action(
                decision.action,
                gaps=gaps_before,
                available_tools=available_tools,
            )
            forced_action_reason = ""
            if action != decision.action:
                forced_action_reason = (
                    f"Tool policy changed model action from {decision.action} to {action} "
                    "to satisfy route evidence requirements."
                )
            if action == "ResearchComplete":
                completed_reason = decision.completion_reason or "research_complete"
                iterations.append(
                    {
                        "iteration": iteration_index + 1,
                        "action": action,
                        "model_action": decision.action,
                        "forced_action_reason": forced_action_reason,
                        "query": None,
                        "new_evidence": 0,
                        "total_evidence": len(evidence),
                        "source_count": self._source_count(evidence),
                        "gaps": gaps_before,
                        "reflection": decision.reflection
                        or self._reflection(
                            item=item,
                            query="ResearchComplete",
                            new_count=0,
                            evidence=evidence,
                            gaps=gaps_before,
                            enough=not gaps_before,
                        ),
                        "rationale": decision.rationale,
                        "next_query": None,
                        "model": usage.model,
                        "tokens_in": usage.prompt_tokens,
                        "tokens_out": usage.completion_tokens,
                    }
                )
                break

            if action == "think_tool":
                next_query = self._queued_or_follow_up_query(
                    item,
                    query_queue,
                    previous_queries,
                    evidence,
                    min_evidence,
                    min_sources,
                )
                follow_up_queries.append(next_query)
                iterations.append(
                    {
                        "iteration": iteration_index + 1,
                        "action": action,
                        "model_action": decision.action,
                        "forced_action_reason": forced_action_reason,
                        "query": None,
                        "new_evidence": 0,
                        "total_evidence": len(evidence),
                        "source_count": self._source_count(evidence),
                        "gaps": gaps_before,
                        "reflection": decision.reflection or "Researcher reflected before the next tool call.",
                        "rationale": decision.rationale,
                        "next_query": next_query,
                        "model": usage.model,
                        "tokens_in": usage.prompt_tokens,
                        "tokens_out": usage.completion_tokens,
                    }
                )
                continue

            if action == "web_search":
                query = self._queued_or_follow_up_query(
                    item,
                    query_queue,
                    previous_queries,
                    evidence,
                    min_evidence,
                    min_sources,
                )
            else:
                query = decision.query or self._queued_or_follow_up_query(
                    item,
                    query_queue,
                    previous_queries,
                    evidence,
                    min_evidence,
                    min_sources,
                )
            previous_queries.append(query)
            before_count = len(evidence)
            tool_start = time.perf_counter()
            normalized_mcp_args = None
            if action == "mcp_tool":
                normalized_mcp_args = self._normalize_mcp_call_args(
                    item,
                    tool_name=decision.mcp_tool_name,
                    tool_args=decision.mcp_tool_args,
                    repository_hint=repository_hint,
                )
                query_evidence = self.collect_mcp(
                    item,
                    query=query,
                    tool_name=decision.mcp_tool_name,
                    tool_args=normalized_mcp_args,
                    repository_hint=repository_hint,
                )
            else:
                query_evidence = self.collect(item, query=query)
            tool_latency_ms = int((time.perf_counter() - tool_start) * 1000)
            evidence = self._dedupe_evidence([*evidence, *query_evidence])
            new_count = len(evidence) - before_count
            gaps = self._sufficiency_gaps(
                evidence,
                min_evidence=min_evidence,
                min_sources=min_sources,
                required_tools=active_required_tools,
            )
            enough = not gaps
            next_query = None
            if not enough and iteration_index + 1 < iteration_budget:
                next_query = self._queued_or_follow_up_query(
                    item,
                    query_queue,
                    previous_queries,
                    evidence,
                    min_evidence,
                    min_sources,
                )
            if not enough and next_query:
                follow_up_queries.append(next_query)
            iterations.append(
                {
                    "iteration": iteration_index + 1,
                    "action": action,
                    "model_action": decision.action,
                    "forced_action_reason": forced_action_reason,
                    "query": query,
                    "tool": action,
                    "mcp_tool_name": decision.mcp_tool_name if action == "mcp_tool" else None,
                    "mcp_tool_args": normalized_mcp_args if action == "mcp_tool" else None,
                    "source_channel": "mcp" if action == "mcp_tool" else "external",
                    "result_count": len(query_evidence),
                    "tool_latency_ms": tool_latency_ms,
                    "new_evidence": new_count,
                    "total_evidence": len(evidence),
                    "source_count": self._source_count(evidence),
                    "gaps": gaps,
                    "reflection": self._reflection(
                        item=item,
                        query=query,
                        new_count=new_count,
                        evidence=evidence,
                        gaps=gaps,
                        enough=enough,
                    ),
                    "model_reflection": decision.reflection,
                    "rationale": decision.rationale,
                    "next_query": next_query,
                    "model": usage.model,
                    "tokens_in": usage.prompt_tokens,
                    "tokens_out": usage.completion_tokens,
                }
            )
            if enough:
                completed_reason = "sufficiency_met"
                break
        else:
            completed_reason = "iteration_limit_reached"

        final_gaps = self._sufficiency_gaps(
            evidence,
            min_evidence=min_evidence,
            min_sources=min_sources,
            required_tools=active_required_tools,
        )
        if not evidence:
            completed_reason = "no_evidence"
        elif final_gaps:
            follow_up_queries.append(self._build_follow_up_query(item, evidence, min_evidence, min_sources))

        return ResearchCollection(
            evidence=evidence,
            iterations=iterations,
            completed_reason=completed_reason,
            follow_up_queries=self._unique(follow_up_queries),
        )

    def _queued_or_follow_up_query(
        self,
        item: PlanItem,
        query_queue: list[str],
        previous_queries: list[str],
        evidence: list[EvidenceItem],
        min_evidence: int,
        min_sources: int,
    ) -> str:
        used = {query.strip().lower() for query in previous_queries}
        for query in query_queue:
            normalized = query.strip()
            if normalized and normalized.lower() not in used:
                return normalized
        query = self._build_follow_up_query(item, evidence, min_evidence, min_sources)
        query_queue.append(query)
        return query

    def _resolve_tool_action(
        self,
        model_action: str,
        *,
        gaps: list[str],
        available_tools: list[str],
    ) -> str:
        if not available_tools:
            return "ResearchComplete" if model_action != "think_tool" else "think_tool"
        gap_text = " ".join(gaps)
        if "needs web_search evidence" in gap_text and "web_search" in available_tools:
            return "web_search"
        if "needs mcp_tool evidence" in gap_text and "mcp_tool" in available_tools:
            return "mcp_tool"
        if model_action in {"ResearchComplete", "think_tool"}:
            return model_action
        if model_action in available_tools:
            return model_action
        return available_tools[0]

    def _sufficiency_gaps(
        self,
        evidence: list[EvidenceItem],
        *,
        min_evidence: int,
        min_sources: int,
        required_tools: list[str] | tuple[str, ...] = (),
    ) -> list[str]:
        gaps: list[str] = []
        if len(evidence) < min_evidence:
            gaps.append(f"needs {min_evidence - len(evidence)} more evidence item(s)")
        source_count = self._source_count(evidence)
        if source_count < min_sources:
            gaps.append(f"needs {min_sources - source_count} more distinct source(s)")
        if "web_search" in required_tools and not self._has_channel_evidence(evidence, "external"):
            gaps.append("needs web_search evidence")
        if "mcp_tool" in required_tools and not self._has_channel_evidence(evidence, "mcp"):
            gaps.append("needs mcp_tool evidence")
        return gaps

    def _has_channel_evidence(self, evidence: list[EvidenceItem], channel: str) -> bool:
        for item in evidence:
            source_channel = str(item.metadata.get("source_channel", ""))
            if source_channel == channel:
                return True
            if channel == "external" and item.kind == "web":
                return True
            if channel == "mcp" and (item.kind == "mcp" or str(item.source).startswith("mcp:")):
                return True
        return False

    def _reflection(
        self,
        *,
        item: PlanItem,
        query: str,
        new_count: int,
        evidence: list[EvidenceItem],
        gaps: list[str],
        enough: bool,
    ) -> str:
        if enough:
            return (
                f"Search/read loop for '{item.id}' found enough evidence after query '{query}'. "
                f"Collected {len(evidence)} item(s) across {self._source_count(evidence)} source(s)."
            )
        gap_text = "; ".join(gaps) if gaps else "evidence is thin"
        return f"Query '{query}' added {new_count} evidence item(s), but the researcher should continue: {gap_text}."

    def _build_follow_up_query(
        self,
        item: PlanItem,
        evidence: list[EvidenceItem],
        min_evidence: int,
        min_sources: int,
    ) -> str:
        source_terms = " ".join(evidence_item.title for evidence_item in evidence[:2] if evidence_item.title)
        base = item.search_query or item.question
        if len(evidence) < min_evidence:
            return f"{base} evidence sources case study"
        if self._source_count(evidence) < min_sources:
            return f"{base} independent source comparison {source_terms}".strip()
        return f"{base} limitations verification"

    def _source_count(self, evidence: list[EvidenceItem]) -> int:
        sources = {
            item.url or f"{item.source}:{item.title}"
            for item in evidence
        }
        return len(sources)

    def _dedupe_evidence(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        deduped: list[EvidenceItem] = []
        seen: set[str] = set()
        for item in evidence:
            key = item.url or f"{item.source}:{item.title}:{item.snippet or item.content or ''}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _unique(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
        return result
