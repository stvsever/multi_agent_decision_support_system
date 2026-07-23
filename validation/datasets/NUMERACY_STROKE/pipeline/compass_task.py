"""COMPASS prediction task for the NUMERACY_STROKE cohort.

Two dissociable numeracy phenotypes are predicted, each as its own univariate
regression on its native population Z-score scale:

    approximate_numeracy  non-symbolic Approximate Number System (dot comparison)
    precise_numeracy      precise symbolic numeracy (WAB number items, writing,
                          dictation, calculation)

They are predicted from the SAME multimodal evidence (demographics, aphasia
severity, whole-brain lesion load, and per-region lesion overlap) but are kept as
separate univariate tasks because the dataset's scientific point is their
DIFFERENTIAL relationship to language and lesion features - approximate numeracy is
largely spared by left-hemisphere language damage while precise numeracy is not.
Running both and comparing per-tier recovery is the readout.

The four COMPASS input files per subject are pre-built by pipeline step 04 under
``compass_inputs/<target_short>_<level>_<cohort>/`` and are reused unmodified here;
only the task spec, the global instruction, and result extraction live in this file.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                 # datasets/NUMERACY_STROKE
REPO_ROOT = ROOT.parents[2]                                # compass_engine
for p in (str(REPO_ROOT), str(REPO_ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

INPUTS_DIR = ROOT / "compass_inputs"
RESULTS_DIR = ROOT / "results"

DATASET_NAME = "NUMERACY_STROKE"
DATASET_LABEL = "Precise vs. Approximate Numeracy in Stroke Participants (OpenNeuro ds006533)"
ONTOLOGY_MODEL = "deepseek/deepseek-v4-flash"

TARGETS = ["approximate_numeracy", "precise_numeracy"]
TARGET_SHORT = {"approximate_numeracy": "approx", "precise_numeracy": "precise"}
TARGET_LABEL = {
    "approximate_numeracy": "Approximate (non-symbolic) numeracy",
    "precise_numeracy": "Precise (symbolic) numeracy",
}
# Data-complexity levels (blinded cohort). The ladder builds up to everything together
# (T3_lesion_fine = demographics + aphasia + per-parcel lesion overlap), then ends with a
# brain-only lesion tier (T4_lesion_brain_only = per-parcel lesion overlap only, no
# demographics/clinical), mirroring the brain-only tail of the other datasets. The coarse
# network-level lesion tier was redundant with the fine map (parcels aggregate to networks).
LEVELS = ["T1_demographics", "T2_aphasia", "T3_lesion_fine", "T4_lesion_brain_only"]

DATASET_CONTEXT = (
    "NUMERACY_STROKE (OpenNeuro ds006533): 105 left-hemisphere chronic stroke survivors. "
    "Both numeracy phenotypes are population Z-scores centred on the stroke cohort (0 = "
    "cohort mean, higher = better numeracy). Predictors come only from the standardized "
    "TRANSFORMED feature tier: raw-year demographics, a standardized WAB-R aphasia quotient "
    "(higher = less aphasia), a standardized log whole-brain lesion fraction, and raw 0-1 "
    "per-region lesion-overlap proportions across cortex (Yeo-7), subcortex (Tian) and "
    "cerebellum (Nettekoven). Key prior: approximate numeracy is largely spared by "
    "left-hemisphere language damage, whereas precise, symbolic numeracy is more vulnerable "
    "to aphasia and peri-sylvian / parietal lesions."
)


def tier_id(target: str, level: str, cohort: str = "blinded") -> str:
    return f"{TARGET_SHORT[target]}_{level}_{cohort}"


def build_task_spec(target: str):
    from src.full_stack.backend.data.models.prediction_task import (
        PredictionMode, PredictionTaskNode, PredictionTaskSpec)
    return PredictionTaskSpec(
        task_id=f"numeracy_{TARGET_SHORT[target]}",
        root=PredictionTaskNode(
            node_id=target,
            display_name=TARGET_LABEL[target],
            mode=PredictionMode.UNIVARIATE_REGRESSION,
            regression_outputs=[target],
            unit_by_output={target: "population Z-score (stroke cohort)"},
        ),
    )


def build_global_instruction(target: str, ref_mean: float, ref_sd: float) -> str:
    other = [t for t in TARGETS if t != target][0]
    return "\n".join([
        DATASET_CONTEXT,
        "",
        f"Predict {target} ({TARGET_LABEL[target]}) for this participant on its native "
        f"population Z-score scale. In the disjoint reference split, mean={ref_mean:.3f}, "
        f"sd={ref_sd:.3f}. Return one numeric value on that scale.",
        f"Do NOT assume {other} is a proxy: the two numeracy systems dissociate after stroke. "
        "Infer the target only from the demographics, aphasia severity, lesion load and "
        "per-region lesion-overlap evidence provided.",
    ])


def reference_stats(target: str) -> tuple[float, float]:
    """Reference mean/sd for a target from the disjoint (non-evaluation) split."""
    import json
    subset = json.loads((RESULTS_DIR / f"subset_{target}.json").read_text())
    df = pd.read_csv(ROOT / "data" / "processed" / "_all_subjects_features.csv")
    eval_src = {p.get("source_participant_id") for p in subset["participants"]}
    ref = df[~df["participant_id"].astype(str).isin({s for s in eval_src if s})]
    s = pd.to_numeric(ref[target], errors="coerce").dropna()
    return float(s.mean()), float(s.std(ddof=0))


def configure_engine(model: str, work_dir: Path) -> None:
    from src.full_stack.backend.config.settings import LLMBackend, get_settings
    s = get_settings()
    s.models.backend = LLMBackend.OPENROUTER
    s.models.public_model_name = model
    for role in ("orchestrator", "critic", "predictor", "integrator", "communicator", "tool"):
        setattr(s.models, f"{role}_model", model)
    s.paths.output_dir = work_dir / "outputs"
    s.paths.logs_dir = work_dir / "logs"
    s.paths.output_dir.mkdir(parents=True, exist_ok=True)
    s.paths.logs_dir.mkdir(parents=True, exist_ok=True)


def harvest_prediction(result: dict, target: str) -> dict:
    pred = (result.get("internal_context") or {}).get("prediction")
    root = getattr(pred, "root_prediction", None)
    out = {"predicted": None}
    if root is None:
        return out
    nodes = [root] + (root.walk() if hasattr(root, "walk") else [])
    for node in nodes:
        reg = getattr(node, "regression", None)
        for k, v in (getattr(reg, "values", None) or {}).items():
            if str(k) == target:
                try:
                    out["predicted"] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def run_engine_on(participant_dir: Path, target: str, global_instruction: str,
                  model: str = ONTOLOGY_MODEL, max_iter: int = 1,
                  work_dir: Path | None = None) -> dict:
    from main import run_compass_pipeline
    work_dir = work_dir or (RESULTS_DIR / "_work" / participant_dir.parent.name / participant_dir.name)
    work_dir.mkdir(parents=True, exist_ok=True)
    configure_engine(model, work_dir)
    spec = build_task_spec(target)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = run_compass_pipeline(
            participant_dir=participant_dir, target_condition=TARGET_LABEL[target],
            control_condition="", prediction_task_spec=spec,
            agent_instructions={"global": global_instruction},
            max_iterations=max_iter, verbose=False, interactive_ui=False)
    return harvest_prediction(result, target)
