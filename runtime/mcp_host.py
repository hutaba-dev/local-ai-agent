"""MCP-aware host and compact tool registry for AHNBYS Phase 1."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock
from time import perf_counter

from mcp import Client

from mcp_servers.academic_server import ACADEMIC_MCP
from mcp_servers.browser_server import BROWSER_MCP
from mcp_servers.context7_server import CONTEXT7_MCP
from mcp_servers.developer_server import DEVELOPER_MCP
from mcp_servers.fetch_server import fetch_mcp
from mcp_servers.github_server import GITHUB_MCP
from mcp_servers.media_server import create_media_mcp
from mcp_servers.project_server import ProjectScope, create_project_mcp
from mcp_servers.search_server import search_mcp
from runtime.capability_registry import CAPABILITIES, TOOL_SPECS
from runtime.projects import ProjectStorageOfflineError


class MCPHealth(str, Enum):
	AVAILABLE = "AVAILABLE"
	UNAVAILABLE = "UNAVAILABLE"
	UNCONFIGURED = "UNCONFIGURED"
	DEGRADED = "DEGRADED"
	RATE_LIMITED = "RATE_LIMITED"
	BUSY = "BUSY"
	OOM = "OOM"
	TIMEOUT = "TIMEOUT"
	MODEL_LIMITED = "MODEL_LIMITED"
	CAPABILITY_LIMITED = "CAPABILITY_LIMITED"
	PROJECT_STORAGE_OFFLINE = "PROJECT_STORAGE_OFFLINE"
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
	if tool_name == "fetch_page":
		return _env_enabled("MCP_FETCH_ENABLED", True)
	spec = TOOL_SPECS.get(tool_name)
	if spec is None:
		return False
	capability = next((item for item in CAPABILITIES if item.name == spec.capability), None)
	if spec.capability == "media":
		return _env_enabled("MCP_MEDIA_ENABLED", _env_enabled("MCP_IMAGE_ENABLED", False))
	default = False if spec.capability == "browser" else True
	return bool(capability and _env_enabled(capability.feature_flag, default))


class MCPHost:
	def __init__(self) -> None:
		self._servers = {
			"search-mcp": search_mcp,
			"web-mcp": fetch_mcp,
			"developer-mcp": DEVELOPER_MCP,
			"context7-mcp": CONTEXT7_MCP,
			"browser-mcp": BROWSER_MCP,
			"academic-mcp": ACADEMIC_MCP,
			"github-mcp": GITHUB_MCP,
		}
		self._tools = {
			name: MCPToolRecord(
				name=spec.name,
				description=spec.description,
				server=spec.server,
				cost=spec.cost_class,
				permission=spec.permission,
				timeout_seconds=180 if spec.server == "media-mcp" else 30 if spec.server in {"browser-mcp", "github-mcp"} else 20,
				input_schema=spec.input_schema,
			)
			for name, spec in TOOL_SPECS.items()
		}
		self._discovered = False
		self._lock = Lock()

	@staticmethod
	def _configured(record: MCPToolRecord, project_scope: ProjectScope | None = None) -> bool:
		if record.server == "browser-mcp":
			return _env_enabled("MCP_PLAYWRIGHT_EGRESS_GUARD", False)
		if record.server == "github-mcp":
			return bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))
		if record.server == "project-mcp":
			return project_scope is not None
		if record.server == "media-mcp":
			from runtime import image_client

			return bool(image_client.IMAGE_WORKER_URL and image_client.IMAGE_WORKER_TOKEN)
		return True

	def _server_for(
		self,
		record: MCPToolRecord,
		project_scope: ProjectScope | None = None,
		media_owner_id: str | None = None,
	) -> object | None:
		if record.server == "project-mcp":
			return create_project_mcp(project_scope) if project_scope is not None else None
		if record.server == "media-mcp":
			if project_scope is not None:
				return create_media_mcp(project_scope)
			from types import SimpleNamespace

			return create_media_mcp(SimpleNamespace(
				owner_id=media_owner_id or f"host-{id(self)}",
				project_id=None, conversation_id=None, tools=None,
			))
		return self._servers.get(record.server)

	async def _discover_async(self) -> None:
		for server_name, server in self._servers.items():
			server_records = [record for record in self._tools.values() if record.server == server_name]
			if server_name == "github-mcp" and not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
				for record in server_records:
					record.health = MCPHealth.UNCONFIGURED.value
				continue
			try:
				async with Client(server, read_timeout_seconds=5) as client:
					result = await client.list_tools()
				discovered = {tool.name: tool for tool in result.tools}
				for record in server_records:
					tool = discovered.get(record.name)
					if tool is None:
						record.health = MCPHealth.UNAVAILABLE.value
						continue
					record.description = tool.description or record.description
					record.input_schema = tool.input_schema
					record.health = MCPHealth.AVAILABLE.value
			except Exception:
				for record in server_records:
					record.health = MCPHealth.UNAVAILABLE.value

	def discover(self, force: bool = False) -> None:
		with self._lock:
			if self._discovered and not force:
				return
			asyncio.run(self._discover_async())
			self._discovered = True

	def catalog(self, project_scope: ProjectScope | None = None) -> list[dict[str, object]]:
		self.discover()
		project_health = MCPHealth.UNCONFIGURED.value
		if project_scope is not None:
			try:
				project_scope.tools.store.get_project(project_scope.owner_id, project_scope.project_id)
				project_scope.tools.store.require_storage()
				project_health = MCPHealth.AVAILABLE.value
			except ProjectStorageOfflineError:
				project_health = MCPHealth.PROJECT_STORAGE_OFFLINE.value
			except Exception:
				project_health = MCPHealth.UNAVAILABLE.value
		catalog = []
		media_health = None
		if any(record.server == "media-mcp" for record in self._tools.values()):
			from runtime.media import MEDIA_DIRECTOR

			media_health = str(MEDIA_DIRECTOR.status().get("health", MCPHealth.UNAVAILABLE.value))
		for record in self._tools.values():
			value = asdict(record)
			if not self._configured(record, project_scope):
				value["health"] = MCPHealth.UNCONFIGURED.value
			elif not mcp_tool_enabled(record.name):
				value["health"] = MCPHealth.UNCONFIGURED.value
			elif record.server == "project-mcp":
				value["health"] = project_health
			elif record.server == "media-mcp" and media_health is not None:
				value["health"] = media_health
			value["available"] = value["health"] == MCPHealth.AVAILABLE.value
			catalog.append(value)
		return catalog

	async def _call_async(
		self,
		tool_name: str,
		arguments: dict[str, object],
		project_scope: ProjectScope | None = None,
		media_owner_id: str | None = None,
	) -> MCPCallOutcome:
		started = perf_counter()
		record = self._tools.get(tool_name)
		if record is None:
			return MCPCallOutcome(
				False, False, tool_name, "", MCPHealth.UNAVAILABLE.value, None,
				"MCP tool is not registered", 0,
			)
		if not self._configured(record, project_scope):
			return MCPCallOutcome(
				False, False, tool_name, record.server, MCPHealth.UNCONFIGURED.value, None,
				"MCP capability is not configured", 0,
			)
		if not mcp_tool_enabled(tool_name):
			return MCPCallOutcome(
				False, False, tool_name, record.server, MCPHealth.UNCONFIGURED.value, None,
				"MCP capability is disabled", 0,
			)
		server = self._server_for(record, project_scope, media_owner_id)
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

	def call(
		self,
		tool_name: str,
		arguments: dict[str, object],
		project_scope: ProjectScope | None = None,
		media_owner_id: str | None = None,
	) -> MCPCallOutcome:
		return asyncio.run(self._call_async(tool_name, arguments, project_scope, media_owner_id))


MCP_HOST = MCPHost()


def mcp_tool_catalog(project_scope: ProjectScope | None = None) -> list[dict[str, object]]:
	return MCP_HOST.catalog(project_scope)


def call_mcp_tool(
	tool_name: str,
	arguments: dict[str, object],
	project_scope: ProjectScope | None = None,
	media_owner_id: str | None = None,
) -> MCPCallOutcome:
	return MCP_HOST.call(tool_name, arguments, project_scope, media_owner_id)
