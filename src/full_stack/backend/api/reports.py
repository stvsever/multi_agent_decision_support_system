"""
Run artifacts: locating them on disk and assembling them for the report views.

The engine writes into one directory per participant, and which directory that
is depends on whether the inputs came from the bundled sample set. This module
resolves that, then reads the artifacts back as one coherent bundle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.settings import get_settings
from .paths import PSEUDO_INPUTS_DIR, PSEUDO_OUTPUTS_DIR

ARTIFACT_LABELS = {
    "performance_report": "Performance report",
    "report_json": "Structured report",
    "report_markdown": "Clinical summary",
    "execution_log": "Execution log",
    "deep_phenotype": "Deep phenotype report",
    "dataflow_audit": "Dataflow audit",
    "xai_report": "Explainability report",
}


def resolve_output_dir(participant_dir: Path, participant_id: str, output_override: str = "") -> Path:
    """Mirror the engine's own output-directory rule so artifacts are found."""
    settings = get_settings()
    root = Path(output_override).expanduser() if output_override else settings.paths.output_dir
    try:
        if os.path.commonpath([participant_dir.resolve(), PSEUDO_INPUTS_DIR.resolve()]) == str(
            PSEUDO_INPUTS_DIR.resolve()
        ):
            root = PSEUDO_OUTPUTS_DIR
    except ValueError:
        pass
    return Path(root) / f"participant_{participant_id}"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, (dict, list)) else None


def artifact_paths(output_dir: Path, participant_id: str) -> Dict[str, Path]:
    return {
        "performance_report": output_dir / f"performance_report_{participant_id}.json",
        "report_json": output_dir / f"report_{participant_id}.json",
        "report_markdown": output_dir / f"report_{participant_id}.md",
        "execution_log": output_dir / f"execution_log_{participant_id}.json",
        "deep_phenotype": output_dir / "deep_phenotype.md",
        "dataflow_audit": output_dir / f"dataflow_audit_{participant_id}.json",
        "xai_report": output_dir / "xai_explainability_report.md",
    }


def list_artifacts(output_dir: Path, participant_id: str) -> List[Dict[str, Any]]:
    rows = []
    for key, path in artifact_paths(output_dir, participant_id).items():
        exists = path.exists()
        rows.append(
            {
                "key": key,
                "label": ARTIFACT_LABELS.get(key, key),
                "file": path.name,
                "path": str(path),
                "exists": exists,
                "size": path.stat().st_size if exists else 0,
                "format": "markdown" if path.suffix == ".md" else "json",
            }
        )
    return rows


def load_bundle(participant_dir: Path, participant_id: str, output_override: str = "") -> Dict[str, Any]:
    """Everything the report views and the PDF renderer need, in one read."""
    output_dir = resolve_output_dir(participant_dir, participant_id, output_override)
    paths = artifact_paths(output_dir, participant_id)
    deep = paths["deep_phenotype"]
    markdown = paths["report_markdown"]
    return {
        "participant_id": participant_id,
        "output_dir": str(output_dir),
        "exists": output_dir.exists(),
        "artifacts": list_artifacts(output_dir, participant_id),
        "performance_report": _read_json(paths["performance_report"]),
        "patient_report": _read_json(paths["report_json"]),
        "execution_log": _read_json(paths["execution_log"]),
        "dataflow_audit": _read_json(paths["dataflow_audit"]),
        "report_markdown": markdown.read_text(errors="replace") if markdown.exists() else "",
        "deep_phenotype_markdown": deep.read_text(errors="replace") if deep.exists() else "",
        "data_overview": _read_json(participant_dir / "data_overview.json"),
        "deviation_map": _read_json(participant_dir / "hierarchical_deviation_map.json"),
    }


def usage_by_model_from_report(performance_report: Optional[Dict[str, Any]], model_hint: str = "") -> Dict[str, Dict[str, int]]:
    """
    Fold the engine's per-component token ledger into per-model counts.

    The ledger records the component, not the model, so a hint is needed when a
    run used a single model for every role.
    """
    usage = (performance_report or {}).get("token_usage") or {}
    calls = usage.get("calls") or []
    if not calls:
        return {}
    bucket: Dict[str, Dict[str, int]] = {}
    for call in calls:
        model = str(call.get("model") or model_hint or "unknown")
        entry = bucket.setdefault(model, {"prompt": 0, "completion": 0, "calls": 0})
        entry["prompt"] += int(call.get("prompt_tokens") or 0)
        entry["completion"] += int(call.get("completion_tokens") or 0)
        entry["calls"] += 1
    return bucket
