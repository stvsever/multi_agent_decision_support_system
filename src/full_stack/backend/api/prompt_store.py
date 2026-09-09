"""
System prompt editing.

The engine loads each agent and tool system prompt from disk when the agent is
constructed, so an edited prompt takes effect on the next run with no restart.
Originals are snapshotted the first time a prompt is edited, which is what makes
"restore default" trustworthy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .paths import AGENT_PROMPTS_DIR, TOOL_PROMPTS_DIR, prompt_overrides_dir

AGENT_PROMPT_ROLES = {
    "orchestrator_prompt.txt": ("orchestrator", "Plans which tools run, in what order, over which domains."),
    "integrator_prompt.txt": ("integrator", "Fuses tool outputs into one evidence representation."),
    "predictor_prompt.txt": ("predictor", "Produces the phenotype outputs and the evidence chain."),
    "critic_prompt.txt": ("critic", "Scores the prediction and decides whether to iterate."),
    "communicator_prompt.txt": ("communicator", "Writes the deep phenotype report."),
}


def _snapshot_path(scope: str, name: str) -> Path:
    return prompt_overrides_dir() / f"{scope}__{name}.original"


def _root(scope: str) -> Path:
    if scope == "agent":
        return AGENT_PROMPTS_DIR
    if scope == "tool":
        return TOOL_PROMPTS_DIR
    raise KeyError(f"Unknown prompt scope: {scope}")


def list_prompts() -> List[Dict[str, Any]]:
    """Every editable prompt, with whether it currently differs from the shipped text."""
    rows: List[Dict[str, Any]] = []
    for scope, root in (("agent", AGENT_PROMPTS_DIR), ("tool", TOOL_PROMPTS_DIR)):
        if not root.exists():
            continue
        for path in sorted(root.glob("*.txt")):
            role, description = AGENT_PROMPT_ROLES.get(path.name, ("tool", ""))
            snapshot = _snapshot_path(scope, path.name)
            content = path.read_text(errors="replace")
            rows.append(
                {
                    "scope": scope,
                    "name": path.name,
                    "role": role if scope == "agent" else path.stem,
                    "title": path.stem.replace("_prompt", "").replace("_", " ").title(),
                    "description": description,
                    "characters": len(content),
                    "lines": content.count("\n") + 1,
                    "modified": snapshot.exists() and snapshot.read_text(errors="replace") != content,
                    "has_snapshot": snapshot.exists(),
                }
            )
    return rows


def read_prompt(scope: str, name: str) -> Dict[str, Any]:
    path = _root(scope) / name
    if not path.is_file() or path.suffix != ".txt":
        raise FileNotFoundError(f"No prompt named {name}")
    snapshot = _snapshot_path(scope, name)
    content = path.read_text(errors="replace")
    return {
        "scope": scope,
        "name": name,
        "content": content,
        "original": snapshot.read_text(errors="replace") if snapshot.exists() else content,
        "modified": snapshot.exists() and snapshot.read_text(errors="replace") != content,
    }


def write_prompt(scope: str, name: str, content: str) -> Dict[str, Any]:
    path = _root(scope) / name
    if not path.is_file() or path.suffix != ".txt":
        raise FileNotFoundError(f"No prompt named {name}")
    snapshot = _snapshot_path(scope, name)
    if not snapshot.exists():
        snapshot.write_text(path.read_text(errors="replace"))
    path.write_text(content)
    return read_prompt(scope, name)


def restore_prompt(scope: str, name: str) -> Dict[str, Any]:
    path = _root(scope) / name
    snapshot = _snapshot_path(scope, name)
    if snapshot.exists():
        path.write_text(snapshot.read_text(errors="replace"))
    return read_prompt(scope, name)
