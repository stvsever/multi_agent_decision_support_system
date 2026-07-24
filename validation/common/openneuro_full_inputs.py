"""Build the full psychosis COMPASS cohort into system-temp scratch.

The repository tracks a small leakage-safe preflight cohort. Full validation needs
the same four engine files for all 143 recordings, but those generated artifacts
should not add thousands of files to git.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

REQUIRED_FILES = (
    "data_overview.json",
    "hierarchical_deviation_map.json",
    "multimodal_data.json",
    "non_numerical_data.txt",
)


def _complete(root: Path, tiers: list[str], participant_ids: list[str]) -> bool:
    return all(
        all((root / tier / participant_id / name).is_file() for name in REQUIRED_FILES)
        for tier in tiers
        for participant_id in participant_ids
    )


def ensure_psychosis_full_inputs(
    repo_root: Path,
    scratch_root: Path,
    *,
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> Path:
    """Return a complete four-tier, 143-recording input root, building if needed."""
    repo_root = Path(repo_root).resolve()
    scratch_root = Path(scratch_root).resolve()
    dataset_root = repo_root / "validation" / "datasets" / "PSYCHOSIS_FIRST_EPISODE"
    out_root = scratch_root / "full_inputs" / "PSYCHOSIS"

    if str(dataset_root) not in sys.path:
        sys.path.insert(0, str(dataset_root))

    from utils import compass_task as task
    from utils import config
    from utils.build_compass_inputs import _dictionary_map, _eeg_names
    from validation.common import compass_writer, deviation
    from validation.common import tiers as tier_tools

    eeg_names = _eeg_names()
    eeg = pd.read_csv(config.RESULTS_ROOT / "eeg_features.csv")
    non_eeg = pd.read_csv(config.RESULTS_ROOT / "non_eeg_features.csv")
    frame = task.build_merged_frame(eeg, non_eeg)
    participant_ids = sorted(frame["recording_id"].astype(str).tolist())
    groups = task.resolve_predictor_groups(non_eeg, eeg_names)
    tiers = task.build_tiers(groups)
    tier_ids = [tier["id"] for tier in tiers]

    if not force and _complete(out_root, tier_ids, participant_ids):
        progress(
            f"[PSYCHOSIS] full inputs ready: {len(participant_ids)} recordings x "
            f"{len(tier_ids)} tiers at {out_root}"
        )
        return out_root

    predictor_cols = sorted({column for tier in tiers for column in tier["columns"]})
    specs = task.build_specs(
        predictor_cols,
        _dictionary_map(),
        set(eeg_names),
    )
    ontology = json.loads((task.ONTOLOGY_DIR / "subclass_structure.json").read_text())
    target_note = task.build_global_instruction(
        task.reference_target_stats(frame, set(participant_ids))
    )
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    for tier in tiers:
        columns = [column for column in tier["columns"] if column in frame.columns]
        projected = tier_tools.project_ontology(ontology, set(columns))
        reference = deviation.ReferenceModel(
            {column: specs[column] for column in columns},
            mode="cohort",
        ).fit(frame[columns])
        tier_root = out_root / tier["id"]
        tier_root.mkdir(parents=True, exist_ok=True)
        progress(
            f"[PSYCHOSIS] building {tier['id']}: {len(participant_ids)} recordings, "
            f"{len(columns)} features"
        )
        for participant_id in participant_ids:
            row = frame.loc[
                frame["recording_id"].astype(str) == participant_id
            ].iloc[0]
            payloads = compass_writer.build_participant_payloads(
                participant_id=participant_id,
                ontology=projected,
                encoded=reference.encode_participant(row),
                target_note=target_note,
                reference_mode="cohort",
            )
            compass_writer.write_participant(tier_root / participant_id, payloads)
        manifest.append(
            {
                "tier": tier["id"],
                "n_recordings": len(participant_ids),
                "n_features": len(columns),
            }
        )

    if not _complete(out_root, tier_ids, participant_ids):
        raise RuntimeError("Psychosis full-input build finished with incomplete folders.")
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "reference": "full predictor cohort; phenotype targets excluded from engine records",
                "tiers": manifest,
            },
            indent=2,
        )
    )
    progress(
        f"[PSYCHOSIS] full input build complete: {len(participant_ids)} recordings x "
        f"{len(tier_ids)} tiers at {out_root}"
    )
    return out_root
