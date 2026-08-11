from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .schemas import MCPToolDescriptor


MCPQueryTool = Callable[..., list[dict[str, object]]]


@dataclass
class MCPToolRegistry:
    server_url: str
    tool_names: list[str]
    transport: str = "streamable_http"
    timeout_seconds: float = 20.0
    auth_required: bool = False
    auth_token: str = ""
    allow_missing: bool = True
    _tool_cache: list[Any] | None = field(default=None, init=False, repr=False)
    _descriptor_cache: list[MCPToolDescriptor] | None = field(default=None, init=False, repr=False)

    def search(
        self,
        query: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> list[dict[str, object]]:
        if not self.server_url:
            return []
        if self.auth_required and not self.auth_token:
            if self.allow_missing:
                return []
            raise RuntimeError("MCP authentication is required but no token is configured.")
        try:
            return asyncio.run(self._search_async(query, tool_name=tool_name, tool_args=tool_args))
        except Exception as exc:
            if self.allow_missing:
                return []
            raise RuntimeError(f"MCP tool registry failed: {exc}") from exc

    def describe_tools(self) -> list[MCPToolDescriptor]:
        if self._descriptor_cache is not None:
            return list(self._descriptor_cache)
        if not self.server_url:
            return []
        if self.auth_required and not self.auth_token:
            if self.allow_missing:
                return []
            raise RuntimeError("MCP authentication is required but no token is configured.")
        try:
            tools = asyncio.run(self._load_tools_async())
        except Exception as exc:
            if self.allow_missing:
                return []
            raise RuntimeError(f"Could not load MCP tool catalog from {self.server_url}: {exc}") from exc
        descriptors = [_descriptor_for_tool(tool) for tool in self._allowed_tools(tools)]
        self._descriptor_cache = descriptors
        return list(descriptors)

    async def _search_async(
        self,
        query: str,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> list[dict[str, object]]:
        tools = await self._load_tools_async()
        selected_tools = self._select_tools(tools, query=query, tool_name=tool_name, tool_args=tool_args)
        evidence: list[dict[str, object]] = []
        for tool in selected_tools:
            selected_name = getattr(tool, "name", "mcp_tool")
            payload = _payload_for_tool(tool, query, tool_args)
            try:
                output = await asyncio.wait_for(
                    self._invoke_tool(tool, payload),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                if self.allow_missing:
                    continue
                raise RuntimeError(f"MCP tool '{selected_name}' failed: {exc}") from exc
            content = _stringify_tool_output(output)
            if not content:
                continue
            evidence.append(
                {
                    "title": f"MCP tool result: {selected_name}",
                    "source": f"mcp:{selected_name}",
                    "kind": "mcp",
                    "snippet": content[:900],
                    "content": content[:4000],
                    "score": 0.74,
                    "metadata": {
                        "mcp_tool_name": selected_name,
                        "mcp_server_url": self.server_url,
                        "mcp_transport": self.transport,
                        "mcp_tool_args": payload,
                        "query": query,
                    },
                }
            )
        return evidence

    async def _load_tools_async(self) -> list[Any]:
        if self._tool_cache is not None:
            return list(self._tool_cache)
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except Exception as exc:
            if self.allow_missing:
                return []
            raise RuntimeError("Install the optional mcp extra to enable MCP tools: pip install -e .[mcp]") from exc

        auth_headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else None
        client = MultiServerMCPClient(
            {
                "server_1": {
                    "url": self._normalized_server_url(),
                    "headers": auth_headers,
                    "transport": self.transport,
                }
            }
        )
        try:
            tools = await asyncio.wait_for(client.get_tools(), timeout=self.timeout_seconds)
        except Exception as exc:
            if self.allow_missing:
                return []
            raise RuntimeError(f"Could not load MCP tools from {self.server_url}: {exc}") from exc
        self._tool_cache = list(tools)
        self._descriptor_cache = None
        return list(self._tool_cache)

    def _allowed_tools(self, tools: list[Any]) -> list[Any]:
        allowed = set(self.tool_names)
        return [
            tool
            for tool in tools
            if not allowed or getattr(tool, "name", "") in allowed
        ]

    def _select_tools(
        self,
        tools: list[Any],
        *,
        query: str,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
    ) -> list[Any]:
        allowed_tools = self._allowed_tools(tools)
        requested = tool_name.strip() if isinstance(tool_name, str) else ""
        if requested:
            matched = [tool for tool in allowed_tools if getattr(tool, "name", "") == requested]
            if matched:
                return matched[:1]
            if self.allow_missing:
                return []
            raise RuntimeError(f"MCP tool '{requested}' is not configured in ARC_MCP_TOOLS.")

        if len(allowed_tools) <= 1:
            return allowed_tools

        selected = _best_tool_for_query(query, tool_args or {}, allowed_tools)
        return [selected] if selected is not None else []

    async def _invoke_tool(self, tool: Any, payload: dict[str, Any]) -> Any:
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(payload)
        coroutine = getattr(tool, "coroutine", None)
        if coroutine is not None:
            return await coroutine(**payload)
        if hasattr(tool, "invoke"):
            return tool.invoke(payload)
        if callable(tool):
            return tool(**payload)
        return None

    def _normalized_server_url(self) -> str:
        url = self.server_url.rstrip("/")
        parsed = urlsplit(url)
        segments = [segment for segment in parsed.path.split("/") if segment]
        if "mcp" in segments:
            return url
        path = f"{parsed.path.rstrip('/')}/mcp" if parsed.path else "/mcp"
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def build_mcp_tool(settings: Any) -> MCPToolRegistry | None:
    if not getattr(settings, "mcp_enabled", False):
        return None
    server_url = getattr(settings, "mcp_server_url", "")
    if not server_url:
        return None
    tool_names = list(getattr(settings, "mcp_tools", []) or [])
    if not tool_names:
        return None
    auth_required = bool(getattr(settings, "mcp_auth_required", False))
    auth_token = getattr(settings, "mcp_auth_token", "")
    if auth_required and not auth_token:
        if getattr(settings, "strict_providers", False):
            raise RuntimeError("ARC_MCP_AUTH_REQUIRED=true but ARC_MCP_AUTH_TOKEN is not configured.")
        return None
    return MCPToolRegistry(
        server_url=server_url,
        tool_names=tool_names,
        transport=getattr(settings, "mcp_transport", "streamable_http"),
        timeout_seconds=float(getattr(settings, "mcp_timeout_seconds", 20.0)),
        auth_required=auth_required,
        auth_token=auth_token,
        allow_missing=not getattr(settings, "strict_providers", False),
    )


def _payload_for_tool(
    tool: Any,
    query: str,
    tool_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    properties, required = _tool_schema_parts(tool)
    cleaned_args = _clean_tool_args(tool_args or {})
    if cleaned_args:
        payload = cleaned_args
        if properties:
            filtered = {key: value for key, value in payload.items() if key in properties}
            payload = filtered or payload
        missing_required = [key for key in required if key not in payload]
        if "query" in missing_required and query:
            payload["query"] = query
        return payload

    for key in ("query", "question", "input", "text", "q"):
        if key in properties:
            return {key: query}
    if len(required) == 1 and isinstance(required[0], str):
        return {required[0]: query}
    if len(properties) == 1:
        only_key = next(iter(properties))
        return {only_key: query}
    return {"query": query}


def _descriptor_for_tool(tool: Any) -> MCPToolDescriptor:
    name = str(getattr(tool, "name", "mcp_tool"))
    description = _trim(str(getattr(tool, "description", "") or ""))
    properties, required = _tool_schema_parts(tool)
    arg_names = list(properties) or list(required)
    optional = [name for name in arg_names if name not in required]
    return MCPToolDescriptor(
        name=name,
        description=description,
        required_args=[str(name) for name in required],
        optional_args=[str(name) for name in optional],
        typical_scenarios=_typical_scenarios(name, description),
    )


def _tool_schema_parts(tool: Any) -> tuple[dict[str, Any], list[str]]:
    schema = getattr(tool, "args_schema", None)
    properties: dict[str, Any] = {}
    required: list[str] = []
    if schema is not None and hasattr(schema, "model_json_schema"):
        try:
            schema_dict = schema.model_json_schema()
            properties = schema_dict.get("properties", {}) if isinstance(schema_dict, dict) else {}
            required = schema_dict.get("required", []) if isinstance(schema_dict, dict) else []
        except Exception:
            properties = {}
            required = []
    return properties, [item for item in required if isinstance(item, str)]


def _best_tool_for_query(
    query: str,
    tool_args: dict[str, Any],
    tools: list[Any],
) -> Any | None:
    scored = [
        (_tool_match_score(tool, query=query, tool_args=tool_args), index, tool)
        for index, tool in enumerate(tools)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_tool = scored[0]
    if best_score > 0:
        return best_tool
    search_like = [
        tool
        for tool in tools
        if str(getattr(tool, "name", "")).startswith(("search", "find"))
    ]
    return search_like[0] if search_like else None


def _tool_match_score(tool: Any, *, query: str, tool_args: dict[str, Any]) -> int:
    name = str(getattr(tool, "name", "")).lower()
    description = str(getattr(tool, "description", "") or "").lower()
    properties, required = _tool_schema_parts(tool)
    arg_names = set(properties) | set(required)
    query_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    score = 0
    for term in query_terms:
        if term and term in name:
            score += 3
        elif term and term in description:
            score += 1

    arg_keys = set(tool_args)
    score += len(arg_keys & arg_names) * 4
    if "query" in arg_names and query:
        score += 2
    if name.startswith(("search", "find")):
        score += 2

    keyword_groups = [
        (("issue", "bug", "failure", "risk"), ("issue", "issues")),
        (("pull", "pr", "merge", "review"), ("pull", "pr")),
        (("release", "version", "changelog"), ("release",)),
        (("code", "source", "implementation", "file", "readme", "path"), ("code", "file", "content")),
        (("repo", "repository", "github", "project"), ("repo", "repository")),
    ]
    query_blob = " ".join(query_terms)
    for query_keywords, name_keywords in keyword_groups:
        if any(keyword in query_blob for keyword in query_keywords) and any(keyword in name for keyword in name_keywords):
            score += 5

    if {"owner", "repo", "path"} <= arg_keys and any(word in name for word in ("file", "content")):
        score += 8
    elif {"owner", "repo"} <= arg_keys and any(word in name for word in ("issue", "pull", "release", "repo")):
        score += 6
    return score


def _clean_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in args.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        cleaned_value = _clean_tool_arg_value(value)
        if cleaned_value in (None, "", [], {}):
            continue
        cleaned[key_text] = cleaned_value
    return cleaned


def _clean_tool_arg_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _clean_tool_args(value)
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            cleaned_item = _clean_tool_arg_value(item)
            if cleaned_item not in (None, "", [], {}):
                cleaned_items.append(cleaned_item)
        return cleaned_items
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value).strip()


def _typical_scenarios(name: str, description: str) -> list[str]:
    lower = " ".join([name, description]).lower()
    if "file" in lower or "content" in lower or "code" in lower:
        return ["Read or inspect repository code, README files, or implementation details."]
    if "issue" in lower:
        return ["Inspect project risks, bugs, open questions, and maintainer discussions."]
    if "pull" in lower or "pr" in lower:
        return ["Inspect pull request activity, review signals, and implementation changes."]
    if "release" in lower:
        return ["Check recent release notes, version changes, and changelog evidence."]
    if "repo" in lower or "repository" in lower:
        return ["Locate or verify GitHub repositories that match a research target."]
    if "search" in lower:
        return ["Search an external MCP-backed evidence source."]
    return ["Call this external MCP tool when its source is more authoritative than web search."]


def _trim(value: str, limit: int = 500) -> str:
    return " ".join(value.split())[:limit]


def _stringify_tool_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    try:
        return json.dumps(output, ensure_ascii=False, indent=2)[:4000]
    except TypeError:
        return str(output).strip()
