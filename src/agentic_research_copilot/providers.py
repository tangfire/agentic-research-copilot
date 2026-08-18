from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Sequence, TypeVar

import httpx
from pydantic import ValidationError

from .provider_base import ModelUsage, ResearchModelProvider
from .provider_validation import ProviderConfigurationError
from .schemas import (
    ChunkContextContract,
    ClarificationContract,
    CorpusProfile,
    EvidenceItem,
    KnowledgeGraphEntity,
    KnowledgeGraphExtractionContract,
    KnowledgeGraphQueryContract,
    KnowledgeGraphRelationship,
    MCPToolDescriptor,
    PlanItem,
    PlannerContract,
    ReporterContract,
    ResearchRequest,
    ResearcherToolDecisionContract,
    ResearchReport,
    ReportSection,
    RetrievalRoute,
    SourceCompressionContract,
    SupervisorDecisionContract,
    VerificationContract,
)


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")
CLARIFICATION_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "for",
    "from",
    "how",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}
VAGUE_RESEARCH_TERMS = {
    "agent",
    "agents",
    "ai",
    "analysis",
    "architecture",
    "copilot",
    "deep",
    "project",
    "rag",
    "research",
    "system",
}

TContract = TypeVar("TContract")


class OpenAICompatibleResearchModelProvider(ResearchModelProvider):
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        chat_model: str,
        embedding_model: str,
        timeout_seconds: float = 30.0,
        temperature: float = 0.2,
        embedding_dimensions: int = 256,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.embedding_dimensions = max(32, embedding_dimensions)

    def clarify_request(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
    ) -> tuple[ClarificationContract, ModelUsage]:
        payload = {
            "request": request.model_dump(),
            "corpus_profile": corpus_profile.model_dump(),
            "instructions": (
                "Follow the Open Deep Research clarify_with_user phase. Decide whether the "
                "user request is specific enough to start research. Ask at most one concise "
                "clarifying question when the scope, target audience, decision context, or "
                "required source type is genuinely missing. If enough information is present, "
                "return a short verification message summarizing the intended research scope."
            ),
        }
        contract, usage = self._chat_structured(
            system_prompt=(
                "You are the clarification gate for an AI Research Copilot. Return valid JSON "
                "only that conforms to the supplied schema. Do not ask unnecessary questions. "
                "Prefer proceeding when the request already contains a concrete topic, target, "
                "comparison, implementation scope, or deliverable."
            ),
            user_payload=payload,
            schema=ClarificationContract.model_json_schema(),
            response_model=ClarificationContract,
        )
        return _normalize_clarification_contract(contract, request), usage

    def decide_researcher_action(
        self,
        *,
        item: PlanItem,
        available_tools: Sequence[str],
        previous_queries: Sequence[str],
        evidence: Sequence[EvidenceItem],
        gaps: Sequence[str],
        iteration: int,
        max_iterations: int,
        mcp_tools: Sequence[MCPToolDescriptor] = (),
    ) -> tuple[ResearcherToolDecisionContract, ModelUsage]:
        payload = {
            "plan_item": item.model_dump(),
            "available_tools": list(available_tools),
            "previous_queries": list(previous_queries),
            "evidence": [
                {
                    "title": item.title,
                    "source": item.source,
                    "kind": item.kind,
                    "url": item.url,
                    "snippet": item.snippet,
                    "score": item.score,
                }
                for item in evidence[:10]
            ],
            "gaps": list(gaps),
            "iteration": iteration,
            "max_iterations": max_iterations,
            "mcp_tools": [tool.model_dump() for tool in mcp_tools],
            "mcp_routing_hints": _mcp_routing_hints(item, mcp_tools),
            "instructions": (
                "Follow the Open Deep Research researcher loop. Choose exactly one next action: "
                "think_tool for reflection, web_search for a new external query, mcp_tool for a "
                "configured MCP tool call, or ResearchComplete when enough evidence has been collected "
                "or the iteration budget is exhausted. Use mcp_tool only when it is listed in "
                "available_tools. When using mcp_tool, choose a tool from the provided MCP catalog, "
                "set mcp_tool_name to that tool name, and provide structured JSON arguments in "
                "mcp_tool_args whenever the tool needs owner/repo/path/issue_number/release/query style "
                "fields. If mcp_routing_hints contains a github_repository target, prefer repository-aware "
                "GitHub MCP tools such as get_file_contents, search_code, list_issues, "
                "list_pull_requests, or get_latest_release with the extracted owner/repo. Keep the "
                "query concrete and source-oriented, but do not cram structured arguments into the "
                "query string."
            ),
        }
        contract, usage = self._chat_structured(
            system_prompt=(
                "You are a focused researcher inside an AI Research Copilot. Return valid JSON "
                "only that conforms to the supplied schema. Be decisive: search when evidence is "
                "thin, use MCP tools when configured and useful, prefer GitHub MCP for repository, "
                "issue, pull request, release, and code evidence, reflect when a pause is needed, "
                "and complete when evidence is sufficient or the budget is exhausted."
            ),
            user_payload=payload,
            schema=ResearcherToolDecisionContract.model_json_schema(),
            response_model=ResearcherToolDecisionContract,
        )
        return _normalize_researcher_action(
            contract,
            item,
            available_tools,
            previous_queries,
            evidence,
            gaps,
            mcp_tools,
        ), usage

    def draft_plan(
        self,
        request: ResearchRequest,
        corpus_profile: CorpusProfile,
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[PlannerContract, ModelUsage]:
        payload = {
            "request": request.model_dump(),
            "corpus_profile": corpus_profile.model_dump(),
            "revision_count": revision_count,
            "revision_notes": list(revision_notes)[:8],
        }
        schema = PlannerContract.model_json_schema()
        return self._chat_structured(
            system_prompt=(
                "You are the planner for a deep research copilot. Your job is to decompose a research topic "
                "into a structured plan of 3-5 focused sub-questions, each independently researchable and "
                "mapping to a distinct section of the final report.\n\n"
                "Guidelines:\n"
                "- Write a clear research_brief that summarizes the goal, approach, and key constraints.\n"
                "- Each plan item must have a specific question (not vague), a clear purpose, and an "
                "optimized search_query tuned for search engines (shorter and keyword-focused).\n"
                "- Avoid overlapping questions. Cover different angles: background, methods, comparisons, "
                "limitations, and practical implications where relevant.\n"
                "- If revision_count > 0, use revision_notes to address previously identified gaps.\n"
                "- If private documents are available (corpus_profile.has_private_docs), include items "
                "that can be grounded in those documents.\n\n"
                "Return valid JSON only that conforms to the supplied schema."
            ),
            user_payload=payload,
            schema=schema,
            response_model=PlannerContract,
        )

    def supervise_research(
        self,
        request: ResearchRequest,
        research_brief: str,
        plan: Sequence[PlanItem],
        retrieval_routes: Sequence[RetrievalRoute],
        corpus_profile: CorpusProfile,
        *,
        revision_count: int = 0,
        revision_notes: Sequence[str] = (),
    ) -> tuple[SupervisorDecisionContract, ModelUsage]:
        payload = {
            "request": request.model_dump(),
            "research_brief": research_brief,
            "plan": [item.model_dump() for item in plan],
            "retrieval_routes": [route.model_dump() for route in retrieval_routes],
            "corpus_profile": corpus_profile.model_dump(),
            "revision_count": revision_count,
            "revision_notes": list(revision_notes)[:8],
            "instructions": (
                "Follow the Open Deep Research supervisor pattern. First reflect with a "
                "think_tool-style call, then use ConductResearch calls to delegate concrete "
                "research units. Use ResearchComplete only as the completion decision after "
                "delegation and verification criteria are clear. Preserve the provided "
                "plan_item_ids; do not invent IDs. Each ConductResearch call must choose "
                "mode, selected_tools, web_queries/internal_queries, min_evidence, "
                "min_sources, and sufficiency_criteria. Prefer primary or official sources "
                "when they are available, keep source quality visible in evaluation, and "
                "treat retrieval_routes as optional candidate hints, not as mandatory final routing decisions."
            ),
        }
        return self._chat_structured(
            system_prompt=(
                "You are the research supervisor for an AI Research Copilot. Return valid JSON only "
                "that conforms to the supplied schema. Emit Open Deep Research-style tool calls: "
                "think_tool for reflection, ConductResearch for delegated research, and "
                "ResearchComplete for completion criteria. ConductResearch must include the evidence "
                "tools and query rewrites needed by the delegated unit. Keep decisions inspectable and "
                "citation-oriented."
            ),
            user_payload=payload,
            schema=SupervisorDecisionContract.model_json_schema(),
            response_model=SupervisorDecisionContract,
        )

    def assess_report(
        self,
        report: ResearchReport,
        evidence: Sequence[EvidenceItem],
        plan: Sequence[PlanItem],
        *,
        revision_count: int = 0,
        max_revisions: int = 2,
    ) -> tuple[VerificationContract, ModelUsage]:
        compact_report = report.model_dump()
        compact_report["summary"] = _trim_text(str(compact_report.get("summary", "")), 900)
        compact_report["sections"] = [
            {
                **section,
                "heading": _trim_text(str(section.get("heading", "")), 180),
                "content": _trim_text(str(section.get("content", "")), 1000),
            }
            for section in compact_report.get("sections", [])[:6]
            if isinstance(section, dict)
        ]
        payload = {
            "report": compact_report,
            "evidence": [
                {
                    "title": _trim_text(item.title, 180),
                    "source": _trim_text(item.source, 140),
                    "kind": item.kind,
                    "url": item.url,
                    "snippet": _trim_text(item.snippet or "", 360),
                    "content": _trim_text(item.content or "", 420),
                    "score": item.score,
                }
                for item in evidence[:12]
            ],
            "plan": [item.model_dump() for item in plan[:12]],
            "revision_count": revision_count,
            "max_revisions": max_revisions,
        }
        return self._chat_structured(
            system_prompt=(
                "You are the verifier for a research copilot. Return valid JSON only "
                "that conforms to the supplied schema."
            ),
            user_payload=payload,
            schema=VerificationContract.model_json_schema(),
            response_model=VerificationContract,
        )

    def compose_report(
        self,
        topic: str,
        sections: Sequence[ReportSection],
        evidence: Sequence[EvidenceItem],
        confidence: float,
    ) -> tuple[ReporterContract, ModelUsage]:
        evidence_index = [
            {
                "index": index,
                "title": _trim_text(item.title, 180),
                "source": _trim_text(item.source, 140),
                "kind": item.kind,
                "url": item.url,
                "snippet": _trim_text(item.snippet or "", 420),
                "content": _trim_text(item.content or "", 520),
                "score": item.score,
            }
            for index, item in enumerate(evidence[:12], start=1)
        ]
        section_drafts = [
            {
                "heading": _trim_text(section.heading, 180),
                "content": _trim_text(section.content, 1200),
                "evidence_count": section.evidence_count,
                "source_summary": [_trim_text(source, 120) for source in section.source_summary[:4]],
                "citation_titles": [
                    _trim_text(citation.title, 180)
                    for citation in section.citations[:6]
                ],
            }
            for section in sections[:6]
        ]
        payload = {
            "topic": _trim_text(topic, 1200),
            "sections": section_drafts,
            "evidence_index": evidence_index,
            "confidence": confidence,
            "instructions": (
                "Synthesize the final report sections from the draft sections and evidence. "
                "Use the same language as the topic/request. Each section must be specific, "
                "citation-backed, and balanced. Use citation_indexes to reference only the "
                "provided evidence_index entries. Do not invent sources, URLs, facts, or citations."
            ),
        }
        return self._chat_structured(
            system_prompt=(
                "You are the final report writer for a deep research copilot, following the "
                "Open Deep Research pattern of synthesizing compressed findings into a "
                "comprehensive citation-backed report. Return valid JSON only that conforms "
                "to the supplied schema. Populate sections with rewritten section drafts and "
                "citation_indexes that map to the provided evidence_index."
            ),
            user_payload=payload,
            schema=ReporterContract.model_json_schema(),
            response_model=ReporterContract,
        )

    def compress_source(
        self,
        *,
        query: str,
        title: str,
        url: str | None,
        raw_content: str,
    ) -> tuple[SourceCompressionContract, ModelUsage]:
        payload = {
            "query": query,
            "source": {
                "title": title,
                "url": url,
            },
            "raw_content": raw_content,
            "instructions": (
                "Compress the source for a downstream research agent. Preserve concrete facts, "
                "numbers, dates, named entities, and any caveats relevant to the query. Do not "
                "invent facts. key_excerpts must be short excerpts or close paraphrases grounded "
                "in the provided raw_content. Set relevance between 0 and 1."
            ),
        }
        return self._chat_structured(
            system_prompt=(
                "You are a source reader for an AI Research Copilot. Return valid JSON only "
                "that conforms to the supplied schema. The output will become citation-backed "
                "evidence, so preserve supportable facts and list limitations when the source "
                "is thin or off-topic."
            ),
            user_payload=payload,
            schema=SourceCompressionContract.model_json_schema(),
            response_model=SourceCompressionContract,
        )

    def contextualize_chunk(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, Any],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> tuple[ChunkContextContract, ModelUsage]:
        payload = {
            "document": {
                "title": document_title,
                "source": source,
                "metadata": _scalar_metadata(metadata, limit=12),
                "excerpt": document_excerpt[:12000],
            },
            "chunk": {
                "index": chunk_index + 1,
                "total_chunks": total_chunks,
                "text": chunk_text[:2400],
            },
            "instructions": (
                "Generate an indexing-time contextual retrieval prefix for this chunk. "
                "The context should be 50-100 tokens, grounded only in the document and chunk, "
                "and explain where the chunk sits in the document, what local topic it covers, "
                "and which concrete entities/terms matter for dense retrieval and BM25. "
                "Do not answer a user question and do not invent facts."
            ),
        }
        contract, usage = self._chat_structured(
            system_prompt=(
                "You create Anthropic-style contextual retrieval prefixes for RAG indexing. "
                "Return valid JSON only that conforms to the supplied schema. The context field "
                "must be one concise paragraph suitable to prepend to a chunk before embedding "
                "and BM25 indexing."
            ),
            user_payload=payload,
            schema=ChunkContextContract.model_json_schema(),
            response_model=ChunkContextContract,
        )
        return _normalize_chunk_context_contract(contract), usage

    def extract_knowledge_graph(
        self,
        *,
        document_title: str,
        source: str,
        metadata: dict[str, Any],
        document_excerpt: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        max_entities: int,
        max_relationships: int,
    ) -> tuple[KnowledgeGraphExtractionContract, ModelUsage]:
        payload = {
            "document": {
                "title": document_title,
                "source": source,
                "metadata": _scalar_metadata(metadata, limit=12),
                "excerpt": document_excerpt[:12000],
            },
            "chunk": {
                "index": chunk_index + 1,
                "total_chunks": total_chunks,
                "text": chunk_text[:4000],
            },
            "limits": {
                "max_entities": max_entities,
                "max_relationships": max_relationships,
            },
            "instructions": (
                "Extract a compact knowledge graph grounded only in this document chunk. "
                "Entities must use stable canonical names, specific entity types, concise factual "
                "descriptions, aliases only when explicitly supported, and confidence from 0 to 1. "
                "Relationships must connect extracted entity names, describe an explicit relation "
                "supported by the chunk, include a short normalized relation_type plus retrieval "
                "keywords, and avoid generic co-occurrence. Prefer fewer high-value records over "
                "speculative records. Do not invent entities or relations."
            ),
        }
        contract, usage = self._chat_structured(
            system_prompt=(
                "You are the indexing-time knowledge graph extractor for a LightRAG-inspired "
                "retrieval system. Return valid JSON only that conforms to the supplied schema. "
                "Extract both entity-level facts and relationship-level facts so downstream "
                "retrieval can support local entity queries and global relationship queries."
            ),
            user_payload=payload,
            schema=KnowledgeGraphExtractionContract.model_json_schema(),
            response_model=KnowledgeGraphExtractionContract,
        )
        return _normalize_knowledge_graph_extraction_contract(
            contract,
            max_entities=max_entities,
            max_relationships=max_relationships,
        ), usage

    def extract_graph_query(
        self,
        *,
        query: str,
        max_local_keywords: int,
        max_global_keywords: int,
    ) -> tuple[KnowledgeGraphQueryContract, ModelUsage]:
        payload = {
            "query": query,
            "limits": {
                "max_local_keywords": max_local_keywords,
                "max_global_keywords": max_global_keywords,
            },
            "instructions": (
                "Split the query into two retrieval views. local_keywords should contain concrete "
                "entity names, components, people, organizations, systems, locations, identifiers, "
                "or narrow concepts. global_keywords should contain relationship types, themes, "
                "actions, mechanisms, risks, causes, effects, or high-level concepts. Keep phrases "
                "short, deduplicated, and grounded in the query."
            ),
        }
        contract, usage = self._chat_structured(
            system_prompt=(
                "You create dual-level graph retrieval keywords for a LightRAG-inspired system. "
                "Return valid JSON only that conforms to the supplied schema. Local keywords target "
                "entities; global keywords target relationships and themes."
            ),
            user_payload=payload,
            schema=KnowledgeGraphQueryContract.model_json_schema(),
            response_model=KnowledgeGraphQueryContract,
        )
        return _normalize_knowledge_graph_query_contract(
            contract,
            max_local_keywords=max_local_keywords,
            max_global_keywords=max_global_keywords,
        ), usage

    def embed_text(self, text: str) -> tuple[list[float], ModelUsage]:
        start = time.perf_counter()
        payload = {"model": self.embedding_model, "input": text, "dimensions": self.embedding_dimensions}
        with self._client() as client:
            response = client.post("/embeddings", json=payload)
            response.raise_for_status()
            body = response.json()
        vector = list(body["data"][0]["embedding"])
        usage = self._usage_from_body(body, start, self.embedding_model)
        return vector, usage

    def embed_texts(self, texts: Sequence[str]) -> tuple[list[list[float]], ModelUsage]:
        start = time.perf_counter()
        payload = {"model": self.embedding_model, "input": list(texts), "dimensions": self.embedding_dimensions}
        with self._client() as client:
            response = client.post("/embeddings", json=payload)
            response.raise_for_status()
            body = response.json()
        vectors = [list(item["embedding"]) for item in body["data"]]
        usage = self._usage_from_body(body, start, self.embedding_model)
        return vectors, usage

    def _chat_structured(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema: dict[str, Any],
        response_model: type[TContract],
    ) -> tuple[TContract, ModelUsage]:
        start = time.perf_counter()
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "output_contract": _compact_schema_contract(schema),
                            "input": user_payload,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
        }
        with self._client() as client:
            last_error: Exception | None = None
            for attempt in range(2):
                if attempt:
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                "The previous response was empty or invalid JSON. "
                                "Return one complete JSON object matching output_contract, with no markdown."
                            ),
                        }
                    )
                response = self._post_chat_completion(client, payload)
                body = response.json()
                content = _extract_chat_content(body)
                try:
                    model = response_model.model_validate_json(_extract_json_object(content))
                except (ValidationError, ValueError) as exc:
                    normalized_content = _normalize_structured_output_content(content, response_model)
                    if normalized_content is not None:
                        try:
                            model = response_model.model_validate_json(normalized_content)
                            usage = self._usage_from_body(body, start, self.chat_model)
                            return model, usage
                        except (ValidationError, ValueError) as normalized_exc:
                            last_error = normalized_exc
                            exc = normalized_exc
                    last_error = exc
                    if content.strip():
                        payload["messages"].append(
                            {
                                "role": "assistant",
                                "content": _trim_text(content, 2000),
                            }
                        )
                    continue
                usage = self._usage_from_body(body, start, self.chat_model)
                return model, usage
        raise ValueError(
            f"OpenAI-compatible provider returned invalid structured output for "
            f"{response_model.__name__}: {last_error}"
        )

    def _post_chat_completion(self, client: httpx.Client, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.post("/chat/completions", json=payload)
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if not _is_retryable_http_error(exc) or attempt == 2:
                    raise
                time.sleep(min(1.5 * (attempt + 1), 4.0))
        raise RuntimeError(f"OpenAI-compatible chat request failed: {last_error}")

    def _client(self) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_seconds,
        )

    def _usage_from_body(self, body: dict[str, Any], start: float, model: str) -> ModelUsage:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ModelUsage(
            provider=self.name,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            cost_usd=0.0,
            latency_ms=latency_ms,
        )


def _normalize_structured_output_content(content: str, response_model: type[Any]) -> str | None:
    text = _extract_json_object(content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if response_model is ResearcherToolDecisionContract:
        payload = dict(payload)
        payload["rationale"] = "" if payload.get("rationale") is None else payload.get("rationale", "")
        payload["reflection"] = "" if payload.get("reflection") is None else payload.get("reflection", "")
        if payload.get("query") is None:
            payload["query"] = None
        if payload.get("mcp_tool_name") is None:
            payload["mcp_tool_name"] = None
        if payload.get("mcp_tool_args") is None:
            payload["mcp_tool_args"] = None
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return None


def build_model_provider(settings: Any) -> ResearchModelProvider:
    if getattr(settings, "model_provider", "openai_compatible") != "openai_compatible":
        raise ProviderConfigurationError(
            "ARC_MODEL_PROVIDER must be openai_compatible. Test fixtures must be injected explicitly."
        )
    return OpenAICompatibleResearchModelProvider(
        base_url=getattr(settings, "model_base_url", ""),
        api_key=getattr(settings, "model_api_key", ""),
        chat_model=getattr(settings, "model_chat_model", "gpt-4o-mini"),
        embedding_model=getattr(settings, "model_embedding_model", "text-embedding-3-small"),
        timeout_seconds=float(getattr(settings, "model_timeout_seconds", 30.0)),
        temperature=float(getattr(settings, "model_temperature", 0.2)),
        embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
    )


def build_embedding_provider(settings: Any, model_provider: ResearchModelProvider | None = None) -> ResearchModelProvider:
    provider_name = getattr(settings, "embedding_provider", "model")
    if provider_name == "openai_compatible":
        return OpenAICompatibleResearchModelProvider(
            base_url=getattr(settings, "embedding_base_url", "") or getattr(settings, "model_base_url", ""),
            api_key=getattr(settings, "embedding_api_key", "") or getattr(settings, "model_api_key", ""),
            chat_model=getattr(settings, "model_chat_model", "gpt-4o-mini"),
            embedding_model=getattr(
                settings,
                "embedding_model",
                getattr(settings, "model_embedding_model", "text-embedding-3-small"),
            ),
            timeout_seconds=float(getattr(settings, "model_timeout_seconds", 30.0)),
            temperature=float(getattr(settings, "model_temperature", 0.2)),
            embedding_dimensions=int(getattr(settings, "embedding_dimensions", 256)),
        )
    if provider_name == "model":
        return model_provider or build_model_provider(settings)
    raise ProviderConfigurationError(
        "ARC_EMBEDDING_PROVIDER must be model or openai_compatible. Test fixtures must be injected explicitly."
    )


def _clarification_missing_dimensions(topic: str) -> list[str]:
    cleaned = _clean_text(topic)
    tokens = [
        token.lower()
        for token in TOKEN_PATTERN.findall(cleaned)
        if token.lower() not in CLARIFICATION_STOPWORDS
    ]
    latin_tokens = [token for token in tokens if re.search(r"[a-zA-Z0-9]", token)]
    is_short = len(cleaned) < 18 and len(tokens) <= 3
    is_generic = bool(latin_tokens) and all(token in VAGUE_RESEARCH_TERMS for token in latin_tokens)
    if not is_short and not is_generic:
        return []

    missing = [
        "the concrete research target or decision you want the report to support",
        "the expected output shape, such as comparison, implementation plan, risk analysis, or interview notes",
    ]
    if len(tokens) <= 2:
        missing.append("any constraints such as timeframe, domain, preferred sources, or technologies")
    return missing


def _normalize_clarification_contract(
    contract: ClarificationContract,
    request: ResearchRequest,
) -> ClarificationContract:
    missing_dimensions = [
        _trim_text(item, 180)
        for item in contract.missing_dimensions
        if _clean_text(item)
    ][:5]
    question = _trim_text(contract.question, 500)
    verification = _trim_text(contract.verification, 500)
    if contract.need_clarification:
        if not question:
            fallback_missing = missing_dimensions or _clarification_missing_dimensions(request.topic)
            question = (
                "Before I start the research, please clarify the scope: "
                + "; ".join(fallback_missing[:3])
                + "."
            )
        verification = ""
    else:
        question = ""
        if not verification:
            verification = (
                f"I have enough information to research '{request.topic}' at {request.depth} depth. "
                "I will build a concrete research brief and gather citation-backed evidence."
            )
        missing_dimensions = []
    return contract.model_copy(
        update={
            "question": question,
            "verification": verification,
            "missing_dimensions": missing_dimensions,
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _researcher_follow_up_query(
    item: PlanItem,
    previous_queries: Sequence[str],
    evidence: Sequence[EvidenceItem],
    gaps: Sequence[str],
) -> str:
    base = item.search_query or item.question
    used = {query.strip().lower() for query in previous_queries if query.strip()}
    source_terms = " ".join(evidence_item.title for evidence_item in evidence[:2] if evidence_item.title)
    candidates = [
        base,
        f"{item.question} {item.purpose} official source evidence",
        f"{base} independent source comparison {source_terms}".strip(),
        f"{base} limitations verification {' '.join(gaps[:2])}".strip(),
    ]
    for candidate in candidates:
        normalized = _clean_text(candidate)
        if normalized and normalized.lower() not in used:
            return normalized
    return _clean_text(f"{base} follow up evidence")


def _normalize_researcher_action(
    contract: ResearcherToolDecisionContract,
    item: PlanItem,
    available_tools: Sequence[str],
    previous_queries: Sequence[str],
    evidence: Sequence[EvidenceItem],
    gaps: Sequence[str],
    mcp_tools: Sequence[MCPToolDescriptor] = (),
) -> ResearcherToolDecisionContract:
    available = set(available_tools)
    action = contract.action
    if action == "mcp_tool" and "mcp_tool" not in available:
        action = "web_search"
    if action == "web_search" and "web_search" not in available:
        action = "ResearchComplete" if evidence else "think_tool"
    if action == "ResearchComplete" and not evidence and gaps:
        action = "web_search" if "web_search" in available else "think_tool"

    query = _trim_text(contract.query or "", 320)
    if action in {"web_search", "mcp_tool"} and not query:
        query = _researcher_follow_up_query(item, previous_queries, evidence, gaps)
    if action not in {"web_search", "mcp_tool"}:
        query = None

    mcp_tool_name = None
    if action == "mcp_tool":
        mcp_tool_name = _trim_text(contract.mcp_tool_name or "", 120) or None
    if action == "mcp_tool" and not mcp_tool_name and len(mcp_tools) == 1:
        mcp_tool_name = _trim_text(mcp_tools[0].name, 120) or None
    mcp_tool_args = _normalize_mcp_tool_args(contract.mcp_tool_args) if action == "mcp_tool" else None

    completion_reason = _trim_text(contract.completion_reason or "", 160) or None
    if action == "ResearchComplete" and completion_reason is None:
        completion_reason = "sufficiency_met" if not gaps and evidence else "research_complete"

    return contract.model_copy(
        update={
            "action": action,
            "query": query,
            "mcp_tool_name": mcp_tool_name,
            "mcp_tool_args": mcp_tool_args,
            "rationale": _trim_text(contract.rationale, 500),
            "reflection": _trim_text(contract.reflection, 700),
            "completion_reason": completion_reason,
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _mcp_routing_hints(item: PlanItem, mcp_tools: Sequence[MCPToolDescriptor]) -> dict[str, Any]:
    if not mcp_tools:
        return {}
    github_target = _github_repository_hints(item)
    if not github_target:
        return {}
    tool_names = {tool.name for tool in mcp_tools}
    suggested_tools = [
        tool_name
        for tool_name in [
            "get_file_contents",
            "search_code",
            "list_issues",
            "search_issues",
            "list_pull_requests",
            "get_latest_release",
        ]
        if tool_name in tool_names
    ]
    return {
        "github_repository": github_target,
        "suggested_tools": suggested_tools,
        "suggested_start": (
            "Use get_file_contents with path='README.md' for repository overview when available, "
            "then search_code/list_issues/list_pull_requests/get_latest_release for implementation, "
            "risk, activity, and release evidence."
        ),
    }


def _github_repository_hints(item: PlanItem) -> dict[str, str] | None:
    text = " ".join(
        part
        for part in [
            item.question,
            item.purpose,
            item.search_query or "",
        ]
        if part
    )
    if not text:
        return None

    url_match = re.search(
        r"github\.com[:/](?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)",
        text,
        flags=re.IGNORECASE,
    )
    if url_match:
        return _clean_github_repo_hint(url_match.group("owner"), url_match.group("repo"))

    lower = text.lower()
    if not any(signal in lower for signal in ("github", "repo", "repository", "仓库", "代码库", "开源项目")):
        return None
    slug_match = re.search(
        r"(?<![A-Za-z0-9_.-])(?P<owner>[A-Za-z0-9][A-Za-z0-9_.-]{0,80})/(?P<repo>[A-Za-z0-9_.-]{1,120})(?![A-Za-z0-9_.-])",
        text,
    )
    if slug_match:
        return _clean_github_repo_hint(slug_match.group("owner"), slug_match.group("repo"))
    return None


def _clean_github_repo_hint(owner: str, repo: str) -> dict[str, str] | None:
    cleaned_owner = owner.strip().strip("/")
    cleaned_repo = repo.strip().strip("/").removesuffix(".git")
    if not cleaned_owner or not cleaned_repo:
        return None
    return {"owner": cleaned_owner, "repo": cleaned_repo}


def _normalize_mcp_tool_args(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value:
        return None
    normalized: dict[str, Any] = {}
    for key, raw_value in value.items():
        key_text = _trim_text(str(key), 80)
        if not key_text:
            continue
        cleaned_value = _normalize_mcp_tool_arg_value(raw_value)
        if cleaned_value is None:
            continue
        normalized[key_text] = cleaned_value
    return normalized or None


def _normalize_mcp_tool_arg_value(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, dict):
        return _normalize_mcp_tool_args(value)
    if isinstance(value, list):
        normalized = [_normalize_mcp_tool_arg_value(item) for item in value]
        normalized = [item for item in normalized if item is not None]
        return normalized or None
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    cleaned = _trim_text(str(value), 240)
    return cleaned or None


def _normalize_chunk_context_contract(contract: ChunkContextContract) -> ChunkContextContract:
    context = _limit_words(contract.context, max_words=110)
    key_terms = [term for term in (_clean_text(term) for term in contract.key_terms) if term][:12]
    return contract.model_copy(
        update={
            "context": context,
            "key_terms": key_terms,
            "provenance_hint": _trim_text(contract.provenance_hint, 220),
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _normalize_knowledge_graph_extraction_contract(
    contract: KnowledgeGraphExtractionContract,
    *,
    max_entities: int,
    max_relationships: int,
) -> KnowledgeGraphExtractionContract:
    entities: list[KnowledgeGraphEntity] = []
    known_names: set[str] = set()
    for entity in contract.entities:
        name = _trim_text(entity.name, 160)
        normalized_name = name.casefold()
        if not name or normalized_name in known_names:
            continue
        known_names.add(normalized_name)
        aliases = [
            alias
            for alias in dict.fromkeys(_trim_text(value, 120) for value in entity.aliases)
            if alias and alias.casefold() != normalized_name
        ][:8]
        entities.append(
            entity.model_copy(
                update={
                    "name": name,
                    "entity_type": _trim_text(entity.entity_type, 80) or "concept",
                    "description": _trim_text(entity.description, 500),
                    "aliases": aliases,
                    "confidence": max(0.0, min(1.0, float(entity.confidence or 0.0))),
                }
            )
        )
        if len(entities) >= max(1, max_entities):
            break

    canonical_names = {entity.name.casefold(): entity.name for entity in entities}
    relationships: list[KnowledgeGraphRelationship] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for relationship in contract.relationships:
        source = canonical_names.get(_clean_text(relationship.source).casefold())
        target = canonical_names.get(_clean_text(relationship.target).casefold())
        relation_type = _trim_text(relationship.relation_type, 100) or "related_to"
        relation_key = (
            (source or "").casefold(),
            (target or "").casefold(),
            relation_type.casefold(),
        )
        if not source or not target or source == target or relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)
        keywords = [
            keyword
            for keyword in dict.fromkeys(_trim_text(value, 100) for value in relationship.keywords)
            if keyword
        ][:10]
        relationships.append(
            relationship.model_copy(
                update={
                    "source": source,
                    "target": target,
                    "relation_type": relation_type,
                    "description": _trim_text(relationship.description, 500),
                    "keywords": keywords,
                    "weight": max(0.05, min(3.0, float(relationship.weight or 1.0))),
                    "confidence": max(0.0, min(1.0, float(relationship.confidence or 0.0))),
                }
            )
        )
        if len(relationships) >= max(1, max_relationships):
            break

    return contract.model_copy(
        update={
            "entities": entities,
            "relationships": relationships,
            "summary": _trim_text(contract.summary, 600),
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _normalize_knowledge_graph_query_contract(
    contract: KnowledgeGraphQueryContract,
    *,
    max_local_keywords: int,
    max_global_keywords: int,
) -> KnowledgeGraphQueryContract:
    local_keywords = [
        keyword
        for keyword in dict.fromkeys(_trim_text(value, 120) for value in contract.local_keywords)
        if keyword
    ][: max(1, max_local_keywords)]
    global_keywords = [
        keyword
        for keyword in dict.fromkeys(_trim_text(value, 120) for value in contract.global_keywords)
        if keyword
    ][: max(1, max_global_keywords)]
    return contract.model_copy(
        update={
            "local_keywords": local_keywords,
            "global_keywords": global_keywords,
            "confidence": max(0.0, min(1.0, float(contract.confidence or 0.0))),
        }
    )


def _scalar_metadata(metadata: dict[str, Any], *, limit: int) -> dict[str, object]:
    scalar: dict[str, object] = {}
    for key, value in sorted(metadata.items()):
        if isinstance(value, (str, int, float, bool)):
            scalar[key] = value
        if len(scalar) >= limit:
            break
    return scalar


def _limit_words(text: str, *, max_words: int) -> str:
    words = _clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,.;:") + "..."


def _extract_chat_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("OpenAI-compatible response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("OpenAI-compatible response did not include message content.")
    return content


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _compact_schema_contract(schema: dict[str, Any]) -> dict[str, Any]:
    definitions = schema.get("$defs", {}) if isinstance(schema.get("$defs"), dict) else {}

    def convert(node: Any, depth: int = 0) -> Any:
        if depth > 8:
            return "object"
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = definitions.get(ref.rsplit("/", 1)[-1], {})
            return convert(target, depth + 1)
        if "anyOf" in node and isinstance(node["anyOf"], list):
            variants = [convert(item, depth + 1) for item in node["anyOf"]]
            return {"anyOf": variants[:4]}
        if "enum" in node:
            return {"enum": node["enum"]}
        node_type = node.get("type")
        if node_type == "object" or "properties" in node:
            properties = node.get("properties", {}) if isinstance(node.get("properties"), dict) else {}
            required = set(node.get("required", []) if isinstance(node.get("required"), list) else [])
            return {
                "type": "object",
                "required": sorted(required),
                "properties": {
                    key: convert(value, depth + 1)
                    for key, value in properties.items()
                },
            }
        if node_type == "array":
            return {"type": "array", "items": convert(node.get("items", {}), depth + 1)}
        if isinstance(node_type, str):
            return {"type": node_type}
        return "value"

    return convert(schema)


def _is_retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code in {408, 409, 425, 429} or status_code >= 500
    return False


def _trim_text(text: str, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
