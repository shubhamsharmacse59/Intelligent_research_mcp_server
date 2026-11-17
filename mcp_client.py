"""
MCP CLIENT - starts server.py and calls its tools over the MCP protocol.

    call_tool("analyze_paper", {"file_name": "attention.pdf"})
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parent / "server.py"


async def _call(name: str, arguments: dict):
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(SERVER.parent))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments, read_timeout_seconds=1800)

    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    if result.is_error:
        raise RuntimeError(text)
    if getattr(result, "structured_content", None):
        data = result.structured_content
        return data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def _list_tools() -> list[str]:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(SERVER.parent))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return [t.name for t in (await session.list_tools()).tools]


def call_tool(name: str, arguments: dict | None = None):
    return asyncio.run(_call(name, arguments or {}))


def list_tools() -> list[str]:
    return asyncio.run(_list_tools())
