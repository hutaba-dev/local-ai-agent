"""Small typed handoff contract between the Main and Coding Agent roles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CodingDelegation:
    target: str
    goal: str
    constraints: tuple[str, ...]
    acceptance_checks: tuple[str, ...]
    commit_requested: bool


def delegate_to_coding(
    *, goal: str, constraints: tuple[str, ...], acceptance_checks: tuple[str, ...], commit_requested: bool
) -> CodingDelegation:
    """Build a complete handoff rather than passing an ambiguous chat request."""
    if not goal.strip():
        raise ValueError("coding delegation requires a goal")
    if not acceptance_checks:
        raise ValueError("coding delegation requires at least one acceptance check")
    return CodingDelegation(
        target="coding",
        goal=goal.strip(),
        constraints=constraints,
        acceptance_checks=acceptance_checks,
        commit_requested=commit_requested,
    )