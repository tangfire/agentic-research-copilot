from __future__ import annotations

import json
import sys

from agentic_research_copilot.mcp_tools import build_mcp_tool
from agentic_research_copilot.settings import (
    GITHUB_MCP_READONLY_TOOLS,
    GITHUB_MCP_READONLY_URL,
    load_settings,
)


def main() -> None:
    base_settings = load_settings()
    settings = base_settings.model_copy(
        update={
            "strict_providers": True,
            "mcp_enabled": True,
            "mcp_server_url": GITHUB_MCP_READONLY_URL,
            "mcp_tools": GITHUB_MCP_READONLY_TOOLS,
            "mcp_auth_required": True,
            "mcp_timeout_seconds": min(15.0, max(5.0, base_settings.mcp_timeout_seconds)),
        }
    )
    try:
        registry = build_mcp_tool(settings)
    except RuntimeError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "server_url": settings.mcp_server_url,
                    "auth_token_configured": bool(settings.mcp_auth_token),
                    "error": str(exc),
                    "required_tokens": [
                        "ARC_MCP_AUTH_TOKEN",
                        "GH_TOKEN",
                        "GITHUB_TOKEN",
                        "GITHUB_PERSONAL_ACCESS_TOKEN",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    if registry is None:
        raise RuntimeError("GitHub MCP registry was not created.")
    descriptors = registry.describe_tools()
    payload = {
        "ok": True,
        "server_url": settings.mcp_server_url,
        "auth_token_configured": bool(settings.mcp_auth_token),
        "configured_tools": settings.mcp_tools,
        "loaded_tool_count": len(descriptors),
        "loaded_tools": [tool.name for tool in descriptors],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
