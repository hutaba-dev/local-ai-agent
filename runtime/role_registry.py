"""Central role metadata for the shared KIM/Qwen brain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from runtime.capability_registry import PermissionClass


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RoleDefinition:
    id: str
    runtime_agent: str
    name: str
    description: str
    instructions_path: str
    preferred_capabilities: tuple[str, ...]
    default_capabilities: tuple[str, ...]
    permission_policy: tuple[PermissionClass, ...]
    memory_policy: str
    output_preferences: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    selectable: bool = True

    def instructions(self) -> str:
        return (REPO_ROOT / self.instructions_path).read_text()


ROLES = (
    RoleDefinition(
        id="secretary",
        runtime_agent="main",
        name="KIM / Secretary",
        description="Central coordinator for conversation, planning, memory, and specialist composition.",
        instructions_path="agents/main/instructions.md",
        preferred_capabilities=("project", "time", "documentation", "web", "browser", "media"),
        default_capabilities=(),
        permission_policy=(PermissionClass.READ,),
        memory_policy="Relevant project memory only; durable writes require an explicit user action.",
        output_preferences=("unified_response", "concise_by_default"),
        aliases=("main",),
    ),
    RoleDefinition(
        id="researcher",
        runtime_agent="research",
        name="KIM / Research",
        description="Evidence gathering, source evaluation, analysis, and cited synthesis.",
        instructions_path="agents/research/instructions.md",
        preferred_capabilities=("web", "academic", "documentation", "project", "media"),
        default_capabilities=(),
        permission_policy=(PermissionClass.READ,),
        memory_policy="Retrieve only relevant project evidence; do not persist conclusions automatically.",
        output_preferences=("source_grounded", "separate_fact_and_inference"),
        aliases=("research",),
    ),
    RoleDefinition(
        id="coder",
        runtime_agent="coding",
        name="KIM / Coding",
        description="Repository inspection, minimal edits, execution, tests, and change review.",
        instructions_path="agents/coding/instructions.md",
        preferred_capabilities=(
            "workspace", "documentation", "git", "github", "web", "browser",
        ),
        default_capabilities=("workspace_read", "workspace_search", "workspace_edit", "execute"),
        permission_policy=(
            PermissionClass.READ,
            PermissionClass.WRITE_WORKSPACE,
            PermissionClass.EXECUTE_SAFE,
        ),
        memory_policy="Retrieve relevant project decisions and constraints; workspace context is not project memory.",
        output_preferences=("actions_and_evidence", "tests", "diff_summary"),
        aliases=("coding",),
    ),
    RoleDefinition(
        id="server",
        runtime_agent="server",
        name="KIM / Server",
        description="Approved infrastructure diagnostics and operations.",
        instructions_path="agents/server/instructions.md",
        preferred_capabilities=("git", "documentation", "web"),
        default_capabilities=(),
        permission_policy=(PermissionClass.READ,),
        memory_policy="Use relevant operational context without automatic durable writes.",
        output_preferences=("validation_evidence", "service_impact"),
        selectable=False,
    ),
    RoleDefinition(
        id="image_director",
        runtime_agent="image",
        name="KIM / Image Director",
        description="Image planning, generation, editing, and quality review.",
        instructions_path="agents/main/instructions.md",
        preferred_capabilities=("media", "web", "project"),
        default_capabilities=("media",),
        permission_policy=(PermissionClass.READ, PermissionClass.WRITE_WORKSPACE),
        memory_policy="Use relevant visual preferences only within the active project scope.",
        output_preferences=("visual_result", "quality_evidence"),
        selectable=False,
    ),
    RoleDefinition(
        id="analyst",
        runtime_agent="research",
        name="KIM / Analyst",
        description="Internal evidence-bounded analysis stage used by Research.",
        instructions_path="agents/research/instructions.md",
        preferred_capabilities=("web", "academic", "project"),
        default_capabilities=(),
        permission_policy=(PermissionClass.READ,),
        memory_policy="No independent durable memory writes.",
        output_preferences=("causal_analysis", "explicit_uncertainty"),
        selectable=False,
    ),
    RoleDefinition(
        id="critic",
        runtime_agent="research",
        name="KIM / Critic",
        description="Internal claim and evidence audit stage used by Research.",
        instructions_path="agents/research/instructions.md",
        preferred_capabilities=(),
        default_capabilities=(),
        permission_policy=(PermissionClass.READ,),
        memory_policy="No independent durable memory writes.",
        output_preferences=("evidence_audit", "identify_unsupported_claims"),
        selectable=False,
    ),
)


ROLE_BY_ID = {role.id: role for role in ROLES}
for role in ROLES:
    for alias in (role.runtime_agent, *role.aliases):
        ROLE_BY_ID.setdefault(alias, role)


CLIENT_DEFAULT_ROLES = {"vscode": "coder", "web": "secretary", "telegram": "secretary"}


def get_role(role_id: str) -> RoleDefinition:
    try:
        return ROLE_BY_ID[role_id]
    except KeyError as error:
        raise ValueError(f"unknown role: {role_id}") from error


def client_default_role(client: str) -> RoleDefinition:
    return get_role(CLIENT_DEFAULT_ROLES.get(client, "secretary"))


def selectable_roles() -> tuple[RoleDefinition, ...]:
    return tuple(role for role in ROLES if role.selectable)


def registry_snapshot() -> list[dict[str, object]]:
    return [asdict(role) for role in ROLES]