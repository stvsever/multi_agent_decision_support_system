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
FEATURE_SPECS = {
    "age": {
        "label": "Age", "stat_type": "numeric", "units": "years",
        "description": "Age of participant in years.",
    },
    "sex": {
        "label": "Biological sex", "stat_type": "binary",
        "description": "Self-reported biological sex (male/female).",
    },
    "handedness": {
        "label": "Handedness", "stat_type": "binary",
        "description": "Left- or right-handed.",
    },
    "BMI": {
        "label": "Body mass index", "stat_type": "numeric", "units": "kg/m^2",
        "invalid_values": [0],
        "description": "Body mass index (kg/m^2). Zero values are data errors and treated as missing.",
    },
    "education_level": {
        "label": "Education level", "stat_type": "ordinal",
        "ordinal_order": ["low", "medium", "high"],
        "description": "Highest completed education level (CBS classification): low/medium/high.",
    },
    "background_SES": {
        "label": "Background socio-economic status", "stat_type": "numeric",
        "description": "Background SES from parental income and education (range 2-6).",
    },
    "BAS_drive": {
        "label": "BAS drive", "stat_type": "numeric",
        "description": "Behavioural Activation System drive subscale.",
    },
    "BAS_fun": {
        "label": "BAS fun-seeking", "stat_type": "numeric",
        "description": "Behavioural Activation System fun-seeking subscale.",
    },
    "BAS_reward": {
        "label": "BAS reward responsiveness", "stat_type": "numeric",
        "description": "Behavioural Activation System reward responsiveness subscale.",
    },
    "BIS": {
        "label": "Behavioural inhibition (BIS)", "stat_type": "numeric",
        "description": "Behavioural Inhibition System scale.",
    },
    "NEO_N": {
        "label": "Neuroticism (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Neuroticism scale (sum of items).",
    },
    "NEO_E": {
        "label": "Extraversion (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Extraversion scale (sum of items).",
    },
    "NEO_O": {
        "label": "Openness (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Openness to experience scale (sum of items).",
    },
    "NEO_A": {
        "label": "Agreeableness (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Agreeableness scale (sum of items).",
    },
    "NEO_C": {
        "label": "Conscientiousness (NEO-FFI)", "stat_type": "numeric",
        "description": "NEO-FFI Conscientiousness scale (sum of items).",
    },
    "STAI_T": {
        "label": "Trait anxiety (STAI-T)", "stat_type": "numeric",
        "description": "State-Trait Anxiety Inventory, trait subscale.",
    },
    "sexual_attraction_M": {
        "label": "Sexual attraction to males", "stat_type": "numeric",
        "description": "Degree of sexual attraction to males (1-7).",
    },
    "sexual_attraction_F": {
        "label": "Sexual attraction to females", "stat_type": "numeric",
        "description": "Degree of sexual attraction to females (1-7).",
    },
    "gender_identity_M": {
        "label": "Gender identity (male)", "stat_type": "numeric",
        "description": "Degree to which the subject felt male (1-7).",
    },
    "gender_identity_F": {
        "label": "Gender identity (female)", "stat_type": "numeric",
        "description": "Degree to which the subject felt female (1-7).",
    },
    "religious_upbringing": {
        "label": "Religious upbringing", "stat_type": "binary",
        "description": "Whether the subject had a religious upbringing (yes/no).",
    },
    "religious_now": {
        "label": "Currently religious", "stat_type": "binary",
        "description": "Whether the subject is religious now (yes/no).",
    },
    "religious_importance": {
        "label": "Importance of religion", "stat_type": "numeric",
        "description": "How much religion plays a role in the subject's life (1-5).",
    },
}

ONTOLOGY_CONTEXT = (
    "AOMIC ID1000: 928 healthy Dutch young adults. Features are self-report "
    "questionnaire scores (NEO-FFI Big Five personality, BIS/BAS reinforcement "
    "sensitivity, STAI trait anxiety), demographics, anthropometrics, "
    "socio-economic status, sexual/gender identity ratings, and religiosity. "
    "Group scales from the same instrument together."
)

# ----------------------------------------------------------- subset run
# Deterministic subset for the live LLM run (kept small for cost with a tiny model).
SUBSET_SIZE = 6
RANDOM_SEED = 42
MAX_ITERATIONS = 1  # actor-critic loop depth for the subset smoke run
