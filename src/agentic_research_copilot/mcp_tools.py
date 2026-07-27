from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


MCPQueryTool = Callable[[str, str | None], list[dict[str, object]]]


@dataclass
class MCPToolRegistry:
    server_url: str
    tool_names: list[str]
    transport: str = "streamable_http"
    timeout_seconds: float = 20.0
    auth_required: bool = False
    auth_token: str = ""
    allow_missing: bool = True

    def search(self, query: str, tool_name: str | None = None) -> list[dict[str, object]]:
        if not self.server_url:
            return []
        if self.auth_required and not self.auth_token:
            if self.allow_missing:
                return []
            raise RuntimeError("MCP authentication is required but no token is configured.")
        try:
            return asyncio.run(self._search_async(query, tool_name=tool_name))
        except Exception as exc:
            if self.allow_missing:
                return []
            raise RuntimeError(f"MCP tool registry failed: {exc}") from exc

    async def _search_async(self, query: str, tool_name: str | None = None) -> list[dict[str, object]]:
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

        allowed = set(self.tool_names)
        requested = tool_name.strip() if isinstance(tool_name, str) else ""
        target = requested if requested and (not allowed or requested in allowed) else ""
        selected_tools = [
            tool
            for tool in tools
            if (not allowed or getattr(tool, "name", "") in allowed)
            and (not target or getattr(tool, "name", "") == target)
        ]
        evidence: list[dict[str, object]] = []
        for tool in selected_tools:
            tool_name = getattr(tool, "name", "mcp_tool")
            try:
                output = await asyncio.wait_for(
                    self._invoke_tool(tool, query),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                if self.allow_missing:
                    continue
                raise RuntimeError(f"MCP tool '{tool_name}' failed: {exc}") from exc
            content = _stringify_tool_output(output)
            if not content:
                continue
            evidence.append(
                {
                    "title": f"MCP tool result: {tool_name}",
                    "source": f"mcp:{tool_name}",
                    "kind": "mcp",
                    "snippet": content[:900],
                    "content": content[:4000],
                    "score": 0.74,
                    "metadata": {
                        "mcp_tool_name": tool_name,
                        "mcp_server_url": self.server_url,
                        "query": query,
                    },
                }
            )
        return evidence

    async def _invoke_tool(self, tool: Any, query: str) -> Any:
        payload = _payload_for_tool(tool, query)
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
        if url.endswith("/mcp"):
            return url
        return f"{url}/mcp"


def build_mcp_tool(settings: Any) -> MCPQueryTool | None:
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
    registry = MCPToolRegistry(
        server_url=server_url,
        tool_names=tool_names,
        transport=getattr(settings, "mcp_transport", "streamable_http"),
        timeout_seconds=float(getattr(settings, "mcp_timeout_seconds", 20.0)),
        auth_required=auth_required,
        auth_token=auth_token,
        allow_missing=not getattr(settings, "strict_providers", False),
    )
    return registry.search


def _payload_for_tool(tool: Any, query: str) -> dict[str, Any]:
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

    for key in ("query", "question", "input", "text", "q"):
        if key in properties:
            return {key: query}
    if len(required) == 1 and isinstance(required[0], str):
        return {required[0]: query}
    if len(properties) == 1:
        only_key = next(iter(properties))
        return {only_key: query}
    return {"query": query}


def _stringify_tool_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    try:
        return json.dumps(output, ensure_ascii=False, indent=2)[:4000]
    except TypeError:
        return str(output).strip()
