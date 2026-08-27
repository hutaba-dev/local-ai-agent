"""MCP-aware host and compact tool registry for AHNBYS Phase 1."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock
from time import perf_counter

from mcp import Client

from mcp_servers.fetch_server import fetch_mcp
from mcp_servers.fetch_server import FETCH_PAGE_DESCRIPTION
from mcp_servers.search_server import SEARCH_NEWS_DESCRIPTION, SEARCH_WEB_DESCRIPTION, search_mcp


class MCPHealth(str, Enum):
	AVAILABLE = "AVAILABLE"
	UNAVAILABLE = "UNAVAILABLE"
	UNCONFIGURED = "UNCONFIGURED"
	DEGRADED = "DEGRADED"
	RATE_LIMITED = "RATE_LIMITED"
	ERROR = "ERROR"


@dataclass
class MCPToolRecord:
	name: str
	description: str
	server: str
	cost: str
	permission: str
	timeout_seconds: int
	health: str = MCPHealth.AVAILABLE.value
	input_schema: dict[str, object] | None = None


@dataclass(frozen=True)
class MCPCallOutcome:
	success: bool
	executed: bool
	tool: str
	server: str
	status: str
	output: dict[str, object] | None
	error: str | None
	duration_ms: int


def _env_enabled(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def mcp_tool_enabled(tool_name: str) -> bool:
	if not _env_enabled("MCP_ENABLED", False):
		return False
	capability_flag = "MCP_FETCH_ENABLED" if tool_name == "fetch_page" else "MCP_SEARCH_ENABLED"
	return _env_enabled(capability_flag, True)


class MCPHost:
	def __init__(self) -> None:
		self._servers = {"search-mcp": search_mcp, "web-mcp": fetch_mcp}
		self._tools = {
			"search_web": MCPToolRecord(
				"search_web", SEARCH_WEB_DESCRIPTION, "search-mcp", "low_variable", "public_network", 20,
			),
			"search_news": MCPToolRecord(
				"search_news", SEARCH_NEWS_DESCRIPTION, "search-mcp", "low_variable", "public_network", 20,
			),
			"fetch_page": MCPToolRecord(
				"fetch_page", FETCH_PAGE_DESCRIPTION, "web-mcp", "very_low", "public_https_only", 15,
			),
		}
		self._discovered = False
		self._lock = Lock()

	async def _discover_async(self) -> None:
		for server_name, server in self._servers.items():
			try:
				async with Client(server, read_timeout_seconds=5) as client:
					result = await client.list_tools()
				discovered = {tool.name: tool for tool in result.tools}
				for record in self._tools.values():
					if record.server != server_name:
						continue
					tool = discovered.get(record.name)
					if tool is None:
						record.health = MCPHealth.UNAVAILABLE.value
						continue
					record.description = tool.description or record.description
					record.input_schema = tool.input_schema
					record.health = MCPHealth.AVAILABLE.value
			except Exception:
				for record in self._tools.values():
					if record.server == server_name:
						record.health = MCPHealth.UNAVAILABLE.value

	def discover(self, force: bool = False) -> None:
		with self._lock:
			if self._discovered and not force:
				return
			asyncio.run(self._discover_async())
			self._discovered = True

	def catalog(self) -> list[dict[str, object]]:
		self.discover()
		catalog = []
		for record in self._tools.values():
			value = asdict(record)
			if not mcp_tool_enabled(record.name):
				value["health"] = MCPHealth.UNCONFIGURED.value
			value["available"] = value["health"] == MCPHealth.AVAILABLE.value
			catalog.append(value)
		return catalog

	async def _call_async(self, tool_name: str, arguments: dict[str, object]) -> MCPCallOutcome:
		started = perf_counter()
		record = self._tools.get(tool_name)
		if record is None:
			return MCPCallOutcome(
				False, False, tool_name, "", MCPHealth.UNAVAILABLE.value, None,
				"MCP tool is not registered", 0,
			)
		server = self._servers.get(record.server)
		if server is None:
			record.health = MCPHealth.UNAVAILABLE.value
			return MCPCallOutcome(
				False, False, tool_name, record.server, record.health, None,
				"MCP server is unavailable", 0,
			)
		invoked = False
		try:
			async with Client(server, read_timeout_seconds=record.timeout_seconds) as client:
				invoked = True
				result = await asyncio.wait_for(
					client.call_tool(tool_name, arguments, read_timeout_seconds=record.timeout_seconds),
					timeout=record.timeout_seconds + 1,
				)
			duration_ms = round((perf_counter() - started) * 1000)
			if result.is_error:
				record.health = MCPHealth.ERROR.value
				return MCPCallOutcome(
					False, True, tool_name, record.server, record.health, None,
					"MCP tool execution failed", duration_ms,
				)
			output = result.structured_content
			if not isinstance(output, dict):
				record.health = MCPHealth.ERROR.value
				return MCPCallOutcome(
					False, True, tool_name, record.server, record.health, None,
					"MCP tool returned an invalid structured result", duration_ms,
				)
			status = str(output.get("status", MCPHealth.AVAILABLE.value))
			record.health = status if status in {item.value for item in MCPHealth} else MCPHealth.AVAILABLE.value
			success = record.health in {MCPHealth.AVAILABLE.value, MCPHealth.DEGRADED.value}
			return MCPCallOutcome(
				success, True, tool_name, record.server, record.health, output,
				None if success else "MCP tool did not return usable data", duration_ms,
			)
		except asyncio.TimeoutError:
			record.health = MCPHealth.DEGRADED.value
			return MCPCallOutcome(
				False, invoked, tool_name, record.server, record.health, None,
				"MCP tool timed out", round((perf_counter() - started) * 1000),
			)
		except Exception:
			record.health = MCPHealth.UNAVAILABLE.value if not invoked else MCPHealth.ERROR.value
			return MCPCallOutcome(
				False, invoked, tool_name, record.server, record.health, None,
				"MCP connection or tool call failed", round((perf_counter() - started) * 1000),
			)

	def call(self, tool_name: str, arguments: dict[str, object]) -> MCPCallOutcome:
		return asyncio.run(self._call_async(tool_name, arguments))


MCP_HOST = MCPHost()


def mcp_tool_catalog() -> list[dict[str, object]]:
	return MCP_HOST.catalog()


def call_mcp_tool(tool_name: str, arguments: dict[str, object]) -> MCPCallOutcome:
	return MCP_HOST.call(tool_name, arguments)
