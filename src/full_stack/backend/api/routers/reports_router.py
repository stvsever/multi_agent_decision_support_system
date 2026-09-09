"""Report artifacts, deep phenotype regeneration, and PDF export."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

from ..config_store import load_config
from ..cost import actual_cost_from_usage
from ..reports import ARTIFACT_LABELS, artifact_paths, load_bundle, resolve_output_dir, usage_by_model_from_report
from ..run_manager import get_run_manager
from ..safe_paths import require_directory

router = APIRouter(prefix="/reports", tags=["reports"])


def _resolve(run_id: Optional[str], participant_dir: Optional[str]) -> Dict[str, Any]:
    manager = get_run_manager()
    if run_id:
        record = manager.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"No run {run_id}")
        bundle = load_bundle(
            record.participant_dir, record.participant_id, record.config.workspace.output_dir
        )
        bundle["run"] = record.summary()
        bundle["cost"] = record.cost
        bundle["usage"] = record.usage
        return bundle
    if participant_dir:
        path = require_directory(participant_dir)
        config = load_config()
        bundle = load_bundle(path, path.name, config.workspace.output_dir)
        usage = usage_by_model_from_report(bundle.get("performance_report"), config.models.default_model)
        bundle["cost"] = actual_cost_from_usage(usage) if usage else None
        return bundle
    raise HTTPException(status_code=400, detail="Provide either run_id or participant_dir.")


@router.get("")
def report_bundle(
    run_id: Optional[str] = Query(None), participant_dir: Optional[str] = Query(None)
) -> Dict[str, Any]:
    return _resolve(run_id, participant_dir)


@router.get("/file", response_class=PlainTextResponse)
def report_file(
    key: str = Query(..., description=f"One of {sorted(ARTIFACT_LABELS)}"),
    run_id: Optional[str] = Query(None),
    participant_dir: Optional[str] = Query(None),
) -> str:
    bundle = _resolve(run_id, participant_dir)
    for artifact in bundle["artifacts"]:
        if artifact["key"] == key:
            if not artifact["exists"]:
                raise HTTPException(status_code=404, detail=f"{artifact['file']} has not been generated.")
            return Path(artifact["path"]).read_text(errors="replace")
    raise HTTPException(status_code=400, detail=f"Unknown artifact key: {key}")


@router.get("/pdf")
def report_pdf(
    run_id: Optional[str] = Query(None), participant_dir: Optional[str] = Query(None)
) -> FileResponse:
    """Render the standardized deep phenotyping PDF."""
    from ..pdf_report import render_run_pdf

    bundle = _resolve(run_id, participant_dir)
    performance = bundle.get("performance_report")
    if not performance:
        raise HTTPException(
            status_code=404,
            detail="No completed run found for this participant. Run the pipeline first.",
        )

    # The id originates from a file on disk, so strip it to a plain name before
    # it reaches a filesystem path or a Content-Disposition header.
    participant_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(bundle["participant_id"]))[:80] or "participant"
    target = Path(tempfile.gettempdir()) / f"compass_{participant_id}_report.pdf"
    render_run_pdf(
        output_path=target,
        performance_report=performance,
        patient_report=bundle.get("patient_report"),
        deep_phenotype_markdown=bundle.get("deep_phenotype_markdown") or "",
        data_overview=bundle.get("data_overview"),
        deviation_map=bundle.get("deviation_map"),
        cost=bundle.get("cost"),
    )
    return FileResponse(
        path=str(target),
        media_type="application/pdf",
        filename=f"COMPASS_{participant_id}_deep_phenotype.pdf",
    )
