"""
Dataset-specific configuration for AOMIC ID1000 (OpenNeuro ds003097).

This is the ONLY place AOMIC-specific knowledge lives. The engine under
``src/full_stack`` and the reusable helpers under ``validation/common`` remain
dataset-agnostic. To validate a new dataset, copy this folder and edit this file.

Prediction task: univariate regression of total intelligence
(``IST_intelligence_total``, the Intelligence Structure Test 2000-R composite)
from non-cognitive phenotype features only. The three IST subscales and the
composite itself are excluded from predictors because they are the target's own
components (circular). DWI scan parameters are excluded as non-phenotypic.
"""

from __future__ import annotations

from pathlib import Path

DATASET_NAME = "AOMIC_ID1000"
DATASET_LABEL = "AOMIC-ID1000 (OpenNeuro ds003097)"
ACCESSION = "ds003097"
LICENSE = "CC0"

ROOT = Path(__file__).resolve().parents[1]          # validation/AOMIC_ID1000
DATASET_DIR = ROOT / "dataset"
ONTOLOGY_DIR = ROOT / "ontology"
INPUTS_DIR = ROOT / "compass_inputs"
RESULTS_DIR = ROOT / "results"
PARTICIPANTS_TSV = DATASET_DIR / "participants.tsv"

# Reference strategy: "auto" picks cohort (n=928 >> 20) -> population-referenced
# z-scores. Set to "absolute" to demonstrate single-subject / no-reference
# ingestion, or "external" with an external_norms mapping.
REFERENCE_MODE = "auto"
EXTERNAL_NORMS: dict | None = None

# Ontology model (small + cheap, via OpenRouter).
ONTOLOGY_MODEL = "google/gemini-3.1-flash-lite"

# ---------------------------------------------------------------- target
TARGET = {
    "column": "IST_intelligence_total",
    "label": "Total Intelligence (IST 2000-R composite)",
    "description": "Total intelligence score from the Intelligence Structure Test 2000-R.",
    "units": "points",
}

# Grounding note passed to all agents so the regressor knows the target scale.
# This is measurement calibration (scale/units), NOT participant-specific leakage.
TARGET_SCALE_NOTE = (
    "Prediction target: IST_intelligence_total, the total score of the Intelligence "
    "Structure Test 2000-R. In this healthy young-adult cohort the score is "
    "approximately Normal with mean ~200 and sd ~40 (observed range 68-296). "
    "Predict the participant's total intelligence as a single number on this scale. "
    "No IST subscale scores are provided; infer from personality, demographic, "
    "motivational, affective, identity and lifestyle features only."
)

# --------------------------------------------------------------- excluded
EXCLUDED_COLUMNS = {
    "participant_id": "identifier",
    "IST_intelligence_total": "target",
    "IST_fluid": "target subscale (circular)",
    "IST_memory": "target subscale (circular)",
    "IST_crystallised": "target subscale (circular)",
    "DWI_TR_run1": "scan acquisition parameter (non-phenotypic)",
    "DWI_TR_run2": "scan acquisition parameter (non-phenotypic)",
    "DWI_TR_run3": "scan acquisition parameter (non-phenotypic)",
}

# ------------------------------------------------------------ predictors
# stat_type: numeric | binary | ordinal | nominal
# invalid_values: numeric sentinels treated as missing (e.g. BMI == 0).
# group: feature-group id used to build complexity tiers.
# subdomain_hint: seeds the ontology builder for clean, interpretable grouping.
TABULAR_FEATURE_SPECS = {
    "age": {
        "label": "Age", "stat_type": "numeric", "units": "years",
        "description": "Age of participant in years.",
        "group": "demographics", "subdomain_hint": "personal_attributes",
    },
    "sex": {
        "label": "Biological sex", "stat_type": "binary",
        "description": "Self-reported biological sex (male/female).",
        "group": "demographics", "subdomain_hint": "personal_attributes",
    },
    "handedness": {
        "label": "Handedness", "stat_type": "binary",
        "description": "Left- or right-handed.",
        "group": "demographics", "subdomain_hint": "personal_attributes",
    },
    "BMI": {
        "label": "Body mass index", "stat_type": "numeric", "units": "kg/m^2",
        "invalid_values": [0],
        "description": "Body mass index (kg/m^2). Zero values are data errors and treated as missing.",
        "group": "demographics", "subdomain_hint": "anthropometrics",
    },
    "education_level": {
        "label": "Education level", "stat_type": "ordinal",
        "ordinal_order": ["low", "medium", "high"],
        "description": "Highest completed education level (CBS classification): low/medium/high.",
        "group": "demographics", "subdomain_hint": "socioeconomic_status",
    },
    "background_SES": {
        "label": "Background socio-economic status", "stat_type": "numeric",
        "description": "Background SES from parental income and education (range 2-6).",
        "group": "demographics", "subdomain_hint": "socioeconomic_status",
    },
    "NEO_N": {
        "label": "Neuroticism (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Neuroticism scale (sum of items).",
        "group": "personality", "subdomain_hint": "big_five_personality",
    },
    "NEO_E": {
        "label": "Extraversion (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Extraversion scale (sum of items).",
        "group": "personality", "subdomain_hint": "big_five_personality",
    },
    "NEO_O": {
        "label": "Openness (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Openness to experience scale (sum of items).",
        "group": "personality", "subdomain_hint": "big_five_personality",
    },
    "NEO_A": {
        "label": "Agreeableness (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Agreeableness scale (sum of items).",
        "group": "personality", "subdomain_hint": "big_five_personality",
    },
    "NEO_C": {
        "label": "Conscientiousness (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Conscientiousness scale (sum of items).",
        "group": "personality", "subdomain_hint": "big_five_personality",
    },
    "BAS_drive": {
        "label": "BAS drive", "stat_type": "numeric",
        "description": "Behavioural Activation System drive subscale.",
        "group": "motivation_affect", "subdomain_hint": "reinforcement_sensitivity",
    },
    "BAS_fun": {
        "label": "BAS fun-seeking", "stat_type": "numeric",
        "description": "Behavioural Activation System fun-seeking subscale.",
        "group": "motivation_affect", "subdomain_hint": "reinforcement_sensitivity",
    },
    "BAS_reward": {
        "label": "BAS reward responsiveness", "stat_type": "numeric",
        "description": "Behavioural Activation System reward responsiveness subscale.",
        "group": "motivation_affect", "subdomain_hint": "reinforcement_sensitivity",
    },
    "BIS": {
        "label": "Behavioural inhibition (BIS)", "stat_type": "numeric",
        "description": "Behavioural Inhibition System scale.",
        "group": "motivation_affect", "subdomain_hint": "reinforcement_sensitivity",
    },
    "STAI_T": {
        "label": "Trait anxiety (STAI-T)", "stat_type": "numeric",
        "description": "State-Trait Anxiety Inventory, trait subscale.",
        "group": "motivation_affect", "subdomain_hint": "anxiety_traits",
    },
    "sexual_attraction_M": {
        "label": "Sexual attraction to males", "stat_type": "numeric",
        "description": "Degree of sexual attraction to males (1-7).",
        "group": "identity_belief", "subdomain_hint": "sexual_and_gender_identity",
    },
    "sexual_attraction_F": {
        "label": "Sexual attraction to females", "stat_type": "numeric",
        "description": "Degree of sexual attraction to females (1-7).",
        "group": "identity_belief", "subdomain_hint": "sexual_and_gender_identity",
    },
    "gender_identity_M": {
        "label": "Gender identity (male)", "stat_type": "numeric",
        "description": "Degree to which the subject felt male (1-7).",
        "group": "identity_belief", "subdomain_hint": "sexual_and_gender_identity",
    },
    "gender_identity_F": {
        "label": "Gender identity (female)", "stat_type": "numeric",
        "description": "Degree to which the subject felt female (1-7).",
        "group": "identity_belief", "subdomain_hint": "sexual_and_gender_identity",
    },
    "religious_upbringing": {
        "label": "Religious upbringing", "stat_type": "binary",
        "description": "Whether the subject had a religious upbringing (yes/no).",
        "group": "identity_belief", "subdomain_hint": "religiosity",
    },
    "religious_now": {
        "label": "Currently religious", "stat_type": "binary",
        "description": "Whether the subject is religious now (yes/no).",
        "group": "identity_belief", "subdomain_hint": "religiosity",
    },
    "religious_importance": {
        "label": "Importance of religion", "stat_type": "numeric",
        "description": "How much religion plays a role in the subject's life (1-5).",
        "group": "identity_belief", "subdomain_hint": "religiosity",
    },
}

# ------------------------------------------------------- brain modality
BRAIN_DIR = ROOT / "brain"
FREESURFER_DIR = BRAIN_DIR / "freesurfer"
CONNECTOME_DIR = BRAIN_DIR / "connectome"
BRAIN_CACHE_DIR = BRAIN_DIR / "_cache"          # raw downloads (gitignored)
# Reference cohort sizes for brain z-scores (include the run subset).
BRAIN_MORPH_REF_SIZE = 120                       # FreeSurfer stats are tiny text files
BRAIN_CONN_REF_SIZE = 24                         # fMRI is heavy; keep just above MIN_COHORT_N

# Group id -> preferred ontology domain label (hint only; projection is by column).
GROUP_DOMAIN = {
    "demographics": "DEMOGRAPHICS_AND_PHYSICAL",
    "personality": "PERSONALITY",
    "motivation_affect": "MOTIVATION_AND_AFFECT",
    "identity_belief": "IDENTITY_AND_BELIEF",
    "brain_morphometry": "BRAIN_MORPHOMETRY",
    "brain_connectome": "BRAIN_CONNECTOME",
}


def _load_brain_specs():
    """Merge in generated brain feature specs if the extraction steps have run."""
    import json
    specs = {}
    for path in (FREESURFER_DIR / "morphometry_specs.json",
                 CONNECTOME_DIR / "connectome_specs.json"):
        if path.exists():
            with open(path) as f:
                specs.update(json.load(f))
    return specs


def all_feature_specs():
    """Full predictor spec set: tabular plus any extracted brain features."""
    specs = dict(TABULAR_FEATURE_SPECS)
    specs.update(_load_brain_specs())
    return specs


def feature_groups():
    """{group_id: [columns]} in a stable order, for tabular and brain groups."""
    order = ["demographics", "personality", "motivation_affect", "identity_belief",
             "brain_morphometry", "brain_connectome"]
    groups = {g: [] for g in order}
    for col, spec in all_feature_specs().items():
        g = spec.get("group", "other")
        groups.setdefault(g, []).append(col)
    return {g: cols for g, cols in groups.items() if cols}


# Backwards-compatible alias used by the single-tier (full-tabular) scripts.
FEATURE_SPECS = TABULAR_FEATURE_SPECS

# ----------------------------------------------------------------- tiers
# Cumulative complexity tiers (each adds a modality), plus brain-only tiers.
# A tier is usable only if every group it names has extracted features.
TIERS = [
    {"id": "T1_demographics", "label": "Tier 1: Demographics",
     "groups": ["demographics"]},
    {"id": "T2_personality", "label": "Tier 2: + Personality (Big Five)",
     "groups": ["demographics", "personality"]},
    {"id": "T3_psychometric", "label": "Tier 3: + Motivation & Affect",
     "groups": ["demographics", "personality", "motivation_affect"]},
    {"id": "T4_identity", "label": "Tier 4: + Identity & Belief (all self-report)",
     "groups": ["demographics", "personality", "motivation_affect", "identity_belief"]},
    {"id": "T5_morphometry", "label": "Tier 5: + Brain morphometry",
     "groups": ["demographics", "personality", "motivation_affect", "identity_belief",
                "brain_morphometry"]},
    {"id": "T6_connectome", "label": "Tier 6: + Brain connectome (full multimodal)",
     "groups": ["demographics", "personality", "motivation_affect", "identity_belief",
                "brain_morphometry", "brain_connectome"]},
    {"id": "B1_morphometry_only", "label": "Brain-only: morphometry",
     "groups": ["brain_morphometry"]},
    {"id": "B2_connectome_only", "label": "Brain-only: connectome",
     "groups": ["brain_connectome"]},
    {"id": "B3_brain_only", "label": "Brain-only: morphometry + connectome",
     "groups": ["brain_morphometry", "brain_connectome"]},
]

ONTOLOGY_CONTEXT = (
    "AOMIC ID1000: 928 healthy Dutch young adults. Features span self-report "
    "questionnaires (NEO-FFI Big Five personality, BIS/BAS reinforcement "
    "sensitivity, STAI trait anxiety), demographics, anthropometrics, "
    "socio-economic status, sexual/gender identity ratings, religiosity, plus "
    "structural brain morphometry (FreeSurfer subcortical volumes and cortical "
    "thickness) and functional connectome (Yeo network connectivity from "
    "movie-watching fMRI). Group measures from the same instrument or brain "
    "system together; keep brain morphometry and connectome as separate domains."
)

# ----------------------------------------------------------- subset run
# Deterministic subset for the live LLM run (kept small for cost with a tiny model).
SUBSET_SIZE = 6
RANDOM_SEED = 42
MAX_ITERATIONS = 1  # actor-critic loop depth for the subset smoke run

# Core columns required to be present for a subject to be eligible for the subset.
CORE_COMPLETE = ["NEO_N", "NEO_E", "NEO_O", "NEO_A", "NEO_C",
                 "BAS_drive", "BIS", "STAI_T", "BMI"]


def select_subset_ids(df):
    """Deterministic run subset: complete core, spread across the IQ range."""
    import numpy as np
    import pandas as pd
    target = TARGET["column"]
    ok = df[pd.to_numeric(df[target], errors="coerce").notna()].copy()
    ok = ok[pd.to_numeric(ok["BMI"], errors="coerce").fillna(0) > 0]
    for col in CORE_COMPLETE:
        ok = ok[pd.to_numeric(ok[col], errors="coerce").notna()]
    ok = ok.assign(_t=pd.to_numeric(ok[target], errors="coerce")).sort_values("_t")
    idx = np.linspace(0, len(ok) - 1, SUBSET_SIZE).round().astype(int)
    return sorted(ok.iloc[idx]["participant_id"].tolist())


def brain_reference_ids(df, n, include=None):
    """Brain z-score reference cohort: the run subset plus additional subjects."""
    include = list(include or [])
    ids = list(include)
    for pid in df["participant_id"].tolist():
        if len(ids) >= n:
            break
        if pid not in ids:
            ids.append(pid)
    return sorted(ids)


def load_merged_frame():
    """Participants table left-joined with any extracted brain feature tables.

    The result has one row per participant and columns for every available feature
    (tabular + morphometry + connectome). Brain columns are NaN for participants
    outside the brain reference cohort, which the encoder treats as missing.
    """
    import pandas as pd
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t", na_values=["n/a", "N/A", ""])
    for csv in (FREESURFER_DIR / "morphometry_features.csv",
                CONNECTOME_DIR / "connectome_features.csv"):
        if csv.exists():
            bdf = pd.read_csv(csv)
            df = df.merge(bdf, on="participant_id", how="left")
    return df


def features_with_hints():
    """Feature dicts carrying structural hints (domain, subdomain) for the ontology."""
    specs = all_feature_specs()
    out = []
    for col, spec in specs.items():
        out.append({
            "id": col,
            "label": spec.get("label", col),
            "definition": spec.get("description", ""),
            "stat_type": spec.get("stat_type", "numeric"),
            "units": spec.get("units"),
            "domain": GROUP_DOMAIN.get(spec.get("group", "other"), "OTHER"),
            "subdomain": spec.get("subdomain_hint", "general"),
        })
    return out
