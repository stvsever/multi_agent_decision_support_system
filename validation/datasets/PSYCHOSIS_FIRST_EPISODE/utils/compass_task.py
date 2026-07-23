"""COMPASS ingestion task for the psychosis EEG cohort (notebook 03).

Defines the single integrated prediction task, the five-tier evidence ladder, the
leakage controls, and the helpers that turn each participant into the engine's four
input files and run the hierarchical prediction.

Task (mixed-type hierarchy):
    diagnosis (binary: Control vs First-Episode Psychosis)
      -> global_severity (BPRS 19-item total, univariate regression)
           -> positive_symptoms (4 SAPS global ratings, multivariate)
           -> negative_symptoms (5 SANS global ratings, multivariate)

Tiers (evidence bundles fed to the engine):
    T1 demographics + socio-economic status
    T2 + full non-neural clinical profile (cognition, IQ, observed functioning)
    T3 + all 836 EEG features (full multimodal ceiling)
    T4 psychosis-implicated lean EEG only (neural floor)
    T5 all 836 EEG only (neural ceiling)

Leakage control: the label, every BPRS/SANS/SAPS item (targets, psychosis-only),
GAS and SFS employment items (control-sparse), and chlorpromazine equivalent
(psychosis-only) are never predictors.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT, RESULTS_ROOT

# Repo root (…/compass_engine) so the engine and shared helpers import cleanly.
REPO_ROOT = ROOT.parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

COMPASS_DIR = RESULTS_ROOT / "compass"
ONTOLOGY_DIR = COMPASS_DIR / "ontology"
INPUTS_DIR = COMPASS_DIR / "inputs"
LADDER_DIR = COMPASS_DIR / "ladder"

DATASET_NAME = "PSYCHOSIS_FEP"
DATASET_LABEL = "First-Episode Psychosis resting EEG (OpenNeuro ds003944 + ds003947)"
ONTOLOGY_MODEL = "google/gemini-3.1-flash-lite"

CONTROL_LABEL = "Control"
CASE_LABEL = "First-Episode Psychosis"

# ---- target columns (resolved against non_eeg_features.csv) ----
BPRS_TOTAL = "target__bprs__total_q01_to_q19"
SAPS_GLOBALS = {
    "SAPS_hallucinations": "target__saps__q07_global_rating_of_hallucinations",
    "SAPS_delusions": "target__saps__q20_global_rating_of_delusions",
    "SAPS_bizarre_behavior": "target__saps__q25_global_rating_of_bizzare_behavior",
    "SAPS_positive_formal_thought_disorder":
        "target__saps__q34_global_rating_of_positive_formal_thought_disorder",
}
SANS_GLOBALS = {
    "SANS_affective_flattening": "target__sans__q07_global_rating_of_affective_flattening",
    "SANS_alogia": "target__sans__q13_global_rating_of_alogia",
    "SANS_avolition_apathy": "target__sans__q17_global_rating_of_avolition",
    "SANS_anhedonia_asociality": "target__sans__q22_anhedonia",
    "SANS_attention": "target__sans__q25_global_rating_of_attention",
}
TARGET_UNITS = {
    BPRS_TOTAL: "BPRS 19-item total (each item 1-7; total 19-133)",
    **{v: "SAPS global rating (0-5)" for v in SAPS_GLOBALS.values()},
    **{v: "SANS global rating (0-5)" for v in SANS_GLOBALS.values()},
}


# --------------------------------------------------------------------------- #
# Prediction task specification
# --------------------------------------------------------------------------- #
def build_task_spec():
    from src.full_stack.backend.data.models.prediction_task import (
        PredictionMode, PredictionTaskNode, PredictionTaskSpec)
    return PredictionTaskSpec(
        task_id="psychosis_diagnosis_and_symptom_profile",
        root=PredictionTaskNode(
            node_id="diagnosis",
            display_name="First-Episode Psychosis vs Control",
            mode=PredictionMode.BINARY_CLASSIFICATION,
            class_labels=[CONTROL_LABEL, CASE_LABEL],
            children=[PredictionTaskNode(
                node_id="global_severity",
                display_name="Overall psychiatric symptom severity (BPRS 19-item total)",
                mode=PredictionMode.UNIVARIATE_REGRESSION,
                regression_outputs=[BPRS_TOTAL],
                unit_by_output={BPRS_TOTAL: TARGET_UNITS[BPRS_TOTAL]},
                children=[
                    PredictionTaskNode(
                        node_id="positive_symptoms",
                        display_name="SAPS positive-symptom global ratings",
                        mode=PredictionMode.MULTIVARIATE_REGRESSION,
                        regression_outputs=list(SAPS_GLOBALS.values()),
                        unit_by_output={v: TARGET_UNITS[v] for v in SAPS_GLOBALS.values()},
                    ),
                    PredictionTaskNode(
                        node_id="negative_symptoms",
                        display_name="SANS negative-symptom global ratings",
                        mode=PredictionMode.MULTIVARIATE_REGRESSION,
                        regression_outputs=list(SANS_GLOBALS.values()),
                        unit_by_output={v: TARGET_UNITS[v] for v in SANS_GLOBALS.values()},
                    ),
                ])]))


ALL_OUTPUTS = [BPRS_TOTAL, *SAPS_GLOBALS.values(), *SANS_GLOBALS.values()]


# --------------------------------------------------------------------------- #
# Predictor resolution
# --------------------------------------------------------------------------- #
MATRICS_DOMAIN_TSCORES = [
    "target__matrics__speedtscr_speed_of_processing_age_and_gender_corrected_t_score",
    "target__matrics__att_vigtscr_attention_vigilance_age_and_gender_corrected_t_score",
    "target__matrics__wmtscr_working_memory_age_and_gender_corrected_t_score",
    "target__matrics__verbtscr_verbal_learning_age_and_gender_corrected_t_score",
    "target__matrics__vistscr_visual_learning_age_and_gender_corrected_t_score",
    "target__matrics__rpstscr_reasoning_and_problem_solving_age_and_gender_corrected_t_score",
    "target__matrics__soccogtscr_social_cognition_age_and_gender_corrected_t_score",
    "target__matrics__overalltscr_overall_composite_age_and_gender_corrected_t_score",
]
WASI_COLS = [
    "target__wasi__two_subtest_full_iq_estimate",
    "target__wasi__vocabulary_t_score",
    "target__wasi__matrix_reasoning_t_score",
]
GAF_FUNCTIONING = [
    "target__functioning__gaf_role_current_0_to_10_higher_better",
    "target__functioning__gaf_role_lifetime_low_0_to_10_higher_better",
    "target__functioning__gaf_role_lifetime_high_0_to_10_higher_better",
    "target__functioning__gaf_social_current_0_to_10_higher_better",
    "target__functioning__gaf_social_lifetime_low_0_to_10_higher_better",
    "target__functioning__gaf_social_lifetime_high_0_to_10_higher_better",
]


def resolve_predictor_groups(non_eeg: pd.DataFrame, eeg_names: list[str]) -> dict[str, list[str]]:
    """Whole-cohort predictor columns by group. Control-sparse columns are excluded
    so that missingness cannot leak the diagnosis label."""
    cols = set(non_eeg.columns)
    controls = non_eeg["target__psychosis__case_control_label"].eq("Control")
    n_control = int(controls.sum())

    def control_coverage(c):
        return int(non_eeg.loc[controls, c].notna().sum())

    demographics = [c for c in ["covariate__demographics__age_years",
                                "covariate__demographics__sex_source_reported",
                                "covariate__demographics__race_source_reported",
                                "covariate__demographics__ethnicity_source_reported"] if c in cols]
    ses = [c for c in non_eeg.columns if c.startswith("covariate__socioeconomic__")]
    cognition = [c for c in (MATRICS_DOMAIN_TSCORES + WASI_COLS) if c in cols]
    functioning = [c for c in GAF_FUNCTIONING if c in cols]
    # Social functioning: numeric SFS items with near-complete control coverage
    # (drops the control-sparse employment block and free-text address items).
    social = []
    for c in non_eeg.columns:
        if not c.startswith("target__social_functioning_scale__"):
            continue
        if re.search(r"sfs_q15", c) or re.search(r"code$", c):
            continue
        s = pd.to_numeric(non_eeg[c], errors="coerce")
        if s.notna().sum() < 100:      # mostly free-text or sparse
            continue
        if control_coverage(c) >= 0.9 * n_control:
            social.append(c)
    return {"demographics": demographics, "ses": ses, "cognition": cognition,
            "functioning": functioning, "social": social,
            "eeg_rich": list(eeg_names), "eeg_lean": lean_eeg_columns(eeg_names)}


def lean_eeg_columns(eeg_names: list[str]) -> list[str]:
    """Psychosis-implicated EEG subset: posterior alpha deficit, frontal/global
    slow-wave excess, slowing ratios, and alpha peak slowing (families A and B)."""
    names = set(eeg_names)
    lean: list[str] = []
    posterior = ["global", "occipital_left", "occipital_right", "parietal_left", "parietal_right"]
    anterior = ["global", "frontal_left", "frontal_right"]
    for measure in ["log10_absolute_power_uv2", "relative_power_fraction_of_1_45_hz"]:
        for scope in posterior:
            lean.append(f"A_spectral__{measure}__alpha_8_13_hz__{scope}")
        for band in ["delta_1_4_hz", "theta_4_8_hz"]:
            for scope in anterior:
                lean.append(f"A_spectral__{measure}__{band}__{scope}")
    for ratio in ["theta_over_alpha", "alpha_over_delta"]:
        lean.append(f"A_spectral__natural_log_power_ratio__{ratio}__global")
    for scope in posterior:
        lean.append(f"B_alpha_peak__center_frequency_hz__{scope}")
    return [c for c in lean if c in names]


def build_tiers(groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    base = groups["demographics"] + groups["ses"]
    clinical = base + groups["cognition"] + groups["functioning"] + groups["social"]
    return [
        {"id": "T1_demographic_socioeconomic",
         "label": "T1: demographics + socio-economic status", "columns": base},
        {"id": "T2_clinical_profile",
         "label": "T2: + cognition, IQ, observed functioning", "columns": clinical},
        {"id": "T3_multimodal_full",
         "label": "T3: + all 836 EEG features (full multimodal)",
         "columns": clinical + groups["eeg_rich"]},
        {"id": "T4_eeg_lean",
         "label": "T4: psychosis-implicated lean EEG only", "columns": groups["eeg_lean"]},
        {"id": "T5_eeg_rich",
         "label": "T5: all 836 EEG only", "columns": groups["eeg_rich"]},
    ]


def excluded_columns(non_eeg: pd.DataFrame) -> dict[str, str]:
    """Columns that must never be predictors, with the reason."""
    excl = {"target__psychosis__case_control_label": "root label",
            "target__psychosis__case_control_binary": "root label",
            "covariate__medication__chlorpromazine_equivalent_at_eeg_scan": "psychosis-only (leaks label)"}
    for c in non_eeg.columns:
        if re.search(r"__(bprs|sans|saps)__", c):
            excl[c] = "symptom target / psychosis-only"
        elif "global_assessment_scale" in c:
            excl[c] = "control-sparse (leaks label)"
        elif re.search(r"sfs_q15", c):
            excl[c] = "employment block, control-sparse (leaks label)"
    return excl


# --------------------------------------------------------------------------- #
# Feature specs and ontology paths
# --------------------------------------------------------------------------- #
_BAND_PRETTY = {"delta_1_4_hz": "Delta (1-4 Hz)", "theta_4_8_hz": "Theta (4-8 Hz)",
                "alpha_8_13_hz": "Alpha (8-13 Hz)", "beta_13_30_hz": "Beta (13-30 Hz)",
                "low_gamma_30_45_hz": "Low gamma (30-45 Hz)"}
_FAMILY_PRETTY = {"A_spectral": "Spectral power", "B_alpha_peak": "Alpha peak",
                  "C_aperiodic": "Aperiodic 1/f", "D_entropy": "Entropy and complexity",
                  "E_fractal": "Fractal", "F_microstates": "Microstates",
                  "G_connectivity": "Functional connectivity", "H_graph": "Graph theory"}


def _pretty(token: str) -> str:
    token = _BAND_PRETTY.get(token, token)
    return token.replace("_", " ").strip().capitalize()


def eeg_ontology_path(col: str) -> list[dict[str, str]]:
    """Deterministic deep ontology path (segment dicts) for an EEG feature column.

    ``path[0]`` is the domain. Each segment is ``{"id", "label"}`` as the ontology
    builder expects; the leaf itself (the column) is appended by the builder.
    """
    parts = col.split("__")
    labels = ["Resting EEG", _FAMILY_PRETTY.get(parts[0], parts[0])]
    labels += [_pretty(tok) for tok in parts[1:-1]]
    segs, acc = [], []
    for lab in labels:
        acc.append(lab)
        segs.append({"id": "__".join(acc), "label": lab})
    return segs


def _seg(sid: str, label: str, definition: str = "") -> dict[str, str]:
    return {"id": sid, "label": label, "definition": definition}


# Abstract domains for the non-neural clinical predictors. Making these explicit
# ``path`` hints (instead of leaving them to the LLM) gives a clean, deep, fully
# reproducible hierarchy and, in particular, breaks the 74-item Social Functioning
# Scale out of one flat list into the instrument's own question blocks.
#
# Ontology structure (2026-07-23): the non-neural clinical predictors sit under two
# primary domains only, Demographics and Socio-economic Status and a single Global
# Functioning umbrella. Global Functioning is the psychiatric superordinate construct
# that gathers, at one secondary level, both the cognitive/intelligence measures
# (MATRICS cognitive domains, WASI IQ) and the social/observed-functioning measures
# (GAF ratings, Social Functioning Scale). Together with the Resting EEG domain that
# leaves three primary nodes: Resting EEG, Demographics + SES, and Global Functioning.
_DEMO_DOM = _seg("DEMOGRAPHICS_AND_SES", "Demographics and Socio-economic Status")
_GLOBAL_FUNC_DOM = _seg(
    "GLOBAL_FUNCTIONING", "Global Functioning",
    "Overall functioning of the participant: cognition and intelligence together with "
    "observed role and social functioning.")

_SFS_BLOCK = [
    (r"sfs_q0?[3-7](_|$)", "sfs_withdrawal_engagement", "Withdrawal and engagement",
     "SFS items on daily routine, time spent alone, initiating contact and leaving home."),
    (r"sfs_q0?[89](_|$)|sfs_q1[01](_|$)", "sfs_interpersonal", "Interpersonal and communication",
     "SFS items on friendships, partnership and the ease of conversation."),
    (r"sfs_q12", "sfs_activities_recreation", "Activities and recreation",
     "SFS block 12: independent activities, hobbies, and recreational or social outings."),
    (r"sfs_q13", "sfs_independence_competence", "Independence and competence",
     "SFS block 13: rated competence at independent-living skills."),
    (r"sfs_q14", "sfs_employment", "Employment", "SFS employment / occupation item."),
]


def clinical_ontology_path(col: str) -> list[dict[str, str]]:
    """Deterministic abstract path for a non-EEG clinical predictor column."""
    if col.startswith("covariate__demographics__"):
        return [_DEMO_DOM, _seg("participant_demographics", "Participant demographics")]
    if col.startswith("covariate__socioeconomic__"):
        return [_DEMO_DOM, _seg("socioeconomic_status", "Socio-economic status (Hollingshead)")]
    # Cognition/intelligence and observed/social functioning both live under the single
    # Global Functioning primary node, as secondary siblings.
    if col.startswith("target__matrics__"):
        return [_GLOBAL_FUNC_DOM, _seg("matrics_cognitive_domains", "MATRICS cognitive domains")]
    if col.startswith("target__wasi__"):
        return [_GLOBAL_FUNC_DOM, _seg("wasi_intelligence", "WASI intelligence estimates")]
    if "global_assessment_of_functioning" in col or col.startswith("target__functioning__gaf"):
        return [_GLOBAL_FUNC_DOM, _seg("gaf_ratings", "Global Assessment of Functioning")]
    if col.startswith("target__social_functioning_scale__"):
        sfs = _seg("social_functioning_scale", "Social Functioning Scale (SFS)")
        for pattern, sid, label, defn in _SFS_BLOCK:
            if re.search(pattern, col):
                return [_GLOBAL_FUNC_DOM, sfs, _seg(sid, label, defn)]
        return [_GLOBAL_FUNC_DOM, sfs, _seg("sfs_other", "Other SFS items")]
    return [_seg("CLINICAL_OTHER", "Other clinical measures")]


def build_specs(columns: list[str], non_eeg_dict: dict[str, dict[str, Any]],
                eeg_names: set[str]) -> dict[str, dict[str, Any]]:
    """Feature specs for ingestion/ontology: label, stat_type, description, path."""
    specs: dict[str, dict[str, Any]] = {}
    nominal = {"covariate__demographics__sex_source_reported",
               "covariate__demographics__race_source_reported",
               "covariate__demographics__ethnicity_source_reported"}
    for col in columns:
        if col in eeg_names:
            parts = col.split("__")
            label = f"{_FAMILY_PRETTY.get(parts[0], parts[0])}: " + " ".join(_pretty(t) for t in parts[1:])
            specs[col] = {"label": label[:120], "stat_type": "numeric",
                          "description": f"Resting-EEG feature {col}.",
                          "source": "resting EEG", "path": eeg_ontology_path(col)}
        else:
            meta = non_eeg_dict.get(col, {})
            specs[col] = {
                "label": meta.get("label", col.split("__")[-1].replace("_", " ").title())[:120],
                "stat_type": "nominal" if col in nominal else "numeric",
                "description": meta.get("description", ""),
                "source": meta.get("source_table", "clinical phenotype"),
                "path": clinical_ontology_path(col),
            }
    return specs


def build_merged_frame(eeg: pd.DataFrame, non_eeg: pd.DataFrame) -> pd.DataFrame:
    """One row per recording with predictors and targets, keyed by recording_id."""
    return non_eeg.merge(eeg, on="recording_id", how="left")


def reference_target_stats(frame: pd.DataFrame, reference_ids: set[str]) -> dict[str, dict[str, float]]:
    ref = frame[frame["recording_id"].isin(reference_ids)]
    stats = {}
    for col in ALL_OUTPUTS:
        s = pd.to_numeric(ref[col], errors="coerce").dropna()
        if len(s):
            stats[col] = {"mean": round(float(s.mean()), 2), "sd": round(float(s.std(ddof=0)), 2),
                          "n": int(len(s))}
    return stats


def build_global_instruction(stats: dict[str, dict[str, float]]) -> str:
    lines = [
        "Cohort: First-Episode Psychosis resting-state EEG study (OpenNeuro ds003944 + ds003947). "
        "Participants are adults early in a psychotic illness plus matched healthy controls; each has a "
        "resting-state EEG recording and a clinical phenotype. Controls have no psychotic disorder and "
        "typically minimal symptoms; cases vary widely in severity.",
        "",
        "Predict whether this resting-EEG participant is a first-episode-psychosis case or a control, "
        "then predict their psychiatric symptom severity on the native clinical scales below.",
        "",
        "Symptom scales (predict on these native ranges; no symptom value for this participant is given):",
        f"- {BPRS_TOTAL}: Brief Psychiatric Rating Scale, sum of 19 items (each 1-7; total 19-133).",
    ]
    if BPRS_TOTAL in stats:
        s = stats[BPRS_TOTAL]
        lines[-1] += f" Reference mean={s['mean']}, sd={s['sd']}."
    for name, col in {**SAPS_GLOBALS, **SANS_GLOBALS}.items():
        line = f"- {col}: {name.replace('_', ' ')}, global rating 0-5."
        if col in stats:
            line += f" Reference mean={stats[col]['mean']}, sd={stats[col]['sd']}."
        lines.append(line)
    lines += [
        "",
        "Controls are expected to have low or absent symptoms; cases vary widely. Infer diagnosis and "
        "severity only from the provided evidence (demographics, socio-economic status, cognition, "
        "observed functioning, and EEG features, depending on the tier).",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Ontology construction (LLM for clinical predictors, path hints for EEG)
# --------------------------------------------------------------------------- #
ONTOLOGY_CONTEXT = (
    "First-episode-psychosis resting-state EEG cohort. Predictors span demographics and socio-economic "
    "status, standardized cognition (MATRICS domains, WASI IQ), observed role and social functioning "
    "(GAF and SFS), and 836 resting-EEG features. The non-neural clinical predictors sit under two "
    "primary domains: Demographics and Socio-economic Status, and a single Global Functioning umbrella "
    "that holds cognition, intelligence, and observed/social functioning together. The EEG features are "
    "placed deterministically under a Resting EEG domain by explicit path and are not part of the "
    "semantic grouping."
)
ONTOLOGY_GUIDANCE = (
    "Organise the non-neural predictors into two domains: (1) Demographics and socio-economic status; "
    "(2) Global functioning, a single umbrella whose secondary nodes are cognition (MATRICS cognitive "
    "domains), intelligence (WASI IQ), Global Assessment of Functioning ratings, and the Social "
    "Functioning Scale blocks. Cognition/intelligence and social/observed functioning are secondary "
    "siblings under Global Functioning, not separate primary domains."
)


def build_ontology(frame: pd.DataFrame, predictor_cols: list[str],
                   non_eeg_dict: dict[str, dict[str, Any]], eeg_names: list[str],
                   llm=None):
    """Build the master ontology over every predictor column."""
    from validation.common import ontology as onto
    from validation.common.ingest import features_for_ontology
    specs = build_specs(predictor_cols, non_eeg_dict, set(eeg_names))
    feats = features_for_ontology(specs, frame)
    tree = onto.build_ontology_tree(feats, DATASET_NAME, ONTOLOGY_CONTEXT,
                                    llm=llm, user_guidance=ONTOLOGY_GUIDANCE)
    return tree, specs


def write_ontology_artifacts(ontology: dict[str, Any], frame: pd.DataFrame,
                             specs: dict[str, dict[str, Any]], reference_ids: set[str],
                             predictor_cols: list[str], llm=None) -> dict[str, Any]:
    """Emit the full linguistic representation of the dataset, mirroring the
    reference pipeline: an automated exploration report, the machine-readable
    subclass JSON, a Protege-loadable OWL file, a hierarchy-encoded benchmark CSV,
    a self-contained interactive HTML ontology explorer, and a QA report (coverage,
    cluster agreement, and an optional LLM MECE review)."""
    from validation.common import explore as expl, ontology as onto, viewer
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    ref = frame[frame["recording_id"].isin(reference_ids)].copy()
    target = "target__psychosis__case_control_binary"

    exploration = None
    try:
        pred_specs = {c: specs[c] for c in predictor_cols if c in specs and c in ref.columns}
        exploration = expl.explore(ref, pred_specs, target=target)
        (ONTOLOGY_DIR / "exploration_report.json").write_text(
            json.dumps(exploration, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # exploration is QA only; never block artifact emission
        (ONTOLOGY_DIR / "exploration_report.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), encoding="utf-8")

    onto.write_subclass_json(ontology, ONTOLOGY_DIR / "subclass_structure.json")
    onto.write_subclass_json(ontology, ONTOLOGY_DIR / "ontology.json")
    onto.write_owl(ontology, ONTOLOGY_DIR / "psychosis_fep.owl")
    viewer.write_viewer(ontology, ONTOLOGY_DIR / "ontology_viewer.html", title=DATASET_LABEL)

    names = onto.hierarchical_names(ontology)
    cols = [c for c in names if c in frame.columns]
    bench = frame[["recording_id", *cols]].rename(columns=names)
    for tgt in [BPRS_TOTAL, *SAPS_GLOBALS.values(), *SANS_GLOBALS.values(), target]:
        if tgt in frame.columns:
            bench["TARGET|" + tgt] = pd.to_numeric(frame[tgt], errors="coerce")
    bench.to_csv(ONTOLOGY_DIR / "ontology_features.csv", index=False)

    report = onto.assess_ontology(ontology, exploration, llm=llm, verify=llm is not None)
    (ONTOLOGY_DIR / "ontology_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def write_tier_inputs(tier: dict[str, Any], eval_ids: list[str], reference_ids: set[str],
                      ontology: dict[str, Any], specs: dict[str, dict[str, Any]],
                      frame: pd.DataFrame, target_note: str) -> tuple[dict, Path]:
    """Write the four COMPASS files for each evaluation subject under one tier.

    Predictor deviations are cohort z-scores referenced to the disjoint reference
    split (all recordings except the evaluation subjects), so a subject is never
    standardized against itself.
    """
    from validation.common import tiers as T, deviation, compass_writer
    cols = [c for c in tier["columns"] if c in frame.columns]
    projected = T.project_ontology(ontology, set(cols))
    ref_specs = {c: specs[c] for c in cols}
    ref = deviation.ReferenceModel(ref_specs, mode="cohort")
    ref.fit(frame[frame["recording_id"].isin(reference_ids)])
    out_root = INPUTS_DIR / tier["id"]
    for rid in eval_ids:
        row = frame[frame["recording_id"] == rid].iloc[0]
        encoded = ref.encode_participant(row)
        payloads = compass_writer.build_participant_payloads(
            participant_id=rid, ontology=projected, encoded=encoded,
            target_note=target_note, reference_mode="cohort")
        compass_writer.write_participant(out_root / rid, payloads)
    return projected, out_root


# --------------------------------------------------------------------------- #
# Engine execution and result harvesting
# --------------------------------------------------------------------------- #
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


def harvest_prediction(result: dict) -> dict:
    """Collect the diagnosis label/probability and every regression value."""
    pred = (result.get("internal_context") or {}).get("prediction")
    root = getattr(pred, "root_prediction", None)
    out: dict[str, Any] = {"diagnosis_label": None, "diagnosis_probability": None,
                           "regression": {}}
    if root is None:
        return out
    nodes = [root] + root.walk() if hasattr(root, "walk") else [root]
    seen = set()
    for node in nodes:
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        cls = getattr(node, "classification", None)
        if cls is not None and out["diagnosis_label"] is None:
            label = (getattr(cls, "predicted_label", None) or getattr(cls, "label", None))
            probs = (getattr(cls, "class_probabilities", None) or getattr(cls, "probabilities", None))
            if label is not None:
                out["diagnosis_label"] = str(label)
            if isinstance(probs, dict):
                out["diagnosis_probability"] = {str(k): float(v) for k, v in probs.items()}
        reg = getattr(node, "regression", None)
        for k, v in (getattr(reg, "values", None) or {}).items():
            try:
                out["regression"][str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def run_engine_on(participant_dir: Path, spec, global_instruction: str,
                  model: str = ONTOLOGY_MODEL, max_iter: int = 1,
                  work_dir: Path | None = None) -> dict:
    import contextlib
    import io as _io
    from main import run_compass_pipeline
    work_dir = work_dir or (LADDER_DIR / "_work" / participant_dir.parent.name / participant_dir.name)
    work_dir.mkdir(parents=True, exist_ok=True)
    configure_engine(model, work_dir)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = run_compass_pipeline(
            participant_dir=participant_dir, target_condition=CASE_LABEL,
            control_condition=CONTROL_LABEL, prediction_task_spec=spec,
            agent_instructions={"global": global_instruction},
            max_iterations=max_iter, verbose=False, interactive_ui=False)
    return harvest_prediction(result)
