"""用于 MCP 客户端集成测试的最小 stdio 服务。"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

server = FastMCP("agentkb-test")


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def echo(text: str) -> dict[str, str]:
    return {"text": text}


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def replace_value(value: str) -> dict[str, str]:
    return {"value": value}


if __name__ == "__main__":
    server.run(transport="stdio")
