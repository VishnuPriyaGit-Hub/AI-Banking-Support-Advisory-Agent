from __future__ import annotations

from app.mcp.server import get_tool_registry


class LocalMCPClient:
    def __init__(self) -> None:
        self.registry = get_tool_registry()

    def call_tool(self, tool_name: str, argument: str) -> str:
        if tool_name not in self.registry:
            raise ValueError(f"Unknown tool: {tool_name}")
        return str(self.registry[tool_name](argument))
