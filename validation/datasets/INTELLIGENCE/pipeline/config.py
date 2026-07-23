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

DATASET_NAME = "INTELLIGENCE"
DATASET_LABEL = "AOMIC-ID1000 (OpenNeuro ds003097)"
ACCESSION = "ds003097"
LICENSE = "CC0"

ROOT = Path(__file__).resolve().parents[1]          # validation/INTELLIGENCE
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

# IST subscales: components of the total. Used ONLY as hierarchical child
# prediction OUTPUTS (multivariate under the total). They are never predictors
# (all three are in EXCLUDED_COLUMNS below), so there is no target leakage.
SUBSCALES = [
    {"column": "IST_fluid", "label": "IST fluid reasoning",
     "description": "Fluid intelligence subscale: reasoning on novel verbal, numerical and figural problems (analogies, number series, matrices)."},
    {"column": "IST_memory", "label": "IST memory",
     "description": "Memory subscale: short-term retention and recall of verbal and figural material."},
    {"column": "IST_crystallised", "label": "IST crystallised knowledge",
     "description": "Crystallised intelligence subscale: acquired verbal and numerical knowledge."},
]

# Static context about the instrument, injected into the engine's global
# instruction so the model interprets the scales correctly.
IST_CONTEXT = (
    "The Intelligence Structure Test 2000-R (IST 2000-R) is a standardized German "
    "psychometric battery of reasoning tasks. All scores here are NATIVE IST points "
    "(sums of correct items), not a conventional 100/15 IQ scale. The total is the "
    "overall composite ability; fluid, memory and crystallised are its components, "
    "so a participant's subscale scores tend to move together with the total."
)

# Cohort context, injected into the engine's global instruction so the model knows the
# population it is predicting for (a healthy general-population sample, not a clinical group).
DATASET_CONTEXT = (
    "Cohort: AOMIC ID1000 (OpenNeuro ds003097), 928 healthy Dutch young adults from the "
    "general population, with no clinical or psychiatric group. Intelligence therefore spans "
    "the normal range; predict within that healthy-population distribution."
)


def target_scale_note(reference_mean: float, reference_sd: float) -> str:
    """Target calibration derived only from the disjoint reference cohort.

    The native IST composite is not an IQ score on a 100 mean, 15 SD scale. The
    prompt contains only aggregate reference-split calibration, never evaluation
    labels, participant identifiers, subscales, or an observed test-set range.
    """
    return (
        "Prediction target: IST_intelligence_total, the native total score of the "
        "Intelligence Structure Test 2000-R. This is not a conventional IQ score. "
        f"In the disjoint reference split, mean={reference_mean:.2f} and "
        f"sd={reference_sd:.2f} native IST points. Predict one numeric value on that "
        "native scale. No IST total or subscale values for this evaluation participant "
        "are provided. Infer only from the provided non-cognitive multimodal features."
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
# subdomain_hint: legacy readability hint (tabular features are grouped semantically
#   by the LLM; brain features instead carry an explicit deep ontology `path`).
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
# The 100-person evaluation is disjoint from these normalization references. Each
# extraction target therefore includes the evaluation set plus at least 20 separate
# reference participants.
BRAIN_MORPH_REF_SIZE = 120                       # FreeSurfer stats are tiny text files
BRAIN_CONN_REF_SIZE = 120                        # 100 evaluation + >=20 reference

# Group id -> preferred ontology domain label (hint only; projection is by column).
GROUP_DOMAIN = {
    "demographics": "DEMOGRAPHICS_AND_PHYSICAL",
    "personality": "PERSONALITY",
    "motivation_affect": "MOTIVATION_AND_AFFECT",
    "identity_belief": "IDENTITY_AND_BELIEF",
    "brain_morphometry": "BRAIN_MORPHOMETRY",
    "brain_connectome": "BRAIN_CONNECTOME",
}

# Measurement source per group. This is factual provenance given to the LLM as
# context for semantic ontology construction; it does NOT force the hierarchy
# (the model proposes the domains/subdomains itself).
SOURCE_INFO = {
    "demographics": "self-report demographics and socio-economic background",
    "personality": "NEO-FFI Big Five personality questionnaire",
    "motivation_affect": "BIS/BAS reinforcement-sensitivity and STAI anxiety questionnaires",
    "identity_belief": "self-report sexual/gender identity and religiosity items",
    "brain_morphometry": "structural T1-weighted MRI morphometry (FreeSurfer)",
    "brain_connectome": "functional connectivity from movie-watching fMRI (Yeo networks)",
}

# Optional free-text guidance for ontology construction. This is the backend hook
# for a UI free-text input: whatever a user types is injected into every
# ontology-building prompt. Here it steers the semantic grouping of the self-report
# features into three clean, stable domains (brain features are placed
# deterministically by path and are unaffected).
ONTOLOGY_USER_GUIDANCE = (
    "Organise the self-report features into exactly three top-level domains: "
    "(1) 'Demographics and Anthropometrics' - age, sex, handedness, BMI, education "
    "and socio-economic background; (2) 'Psychological Constructs' - NEO-FFI Big "
    "Five personality, BIS/BAS reinforcement sensitivity, and STAI trait anxiety, "
    "each as its own subdomain; (3) 'Identity and Belief Systems' - sexual and "
    "gender identity ratings, and religiosity. Keep identity and belief separate "
    "from demographics."
)


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
    {"id": "T2_psychological",
     "label": "Tier 2: + Psychological factors (Big Five, BIS/BAS, STAI, identity, religiosity)",
     "groups": ["demographics", "personality", "motivation_affect", "identity_belief"]},
    {"id": "T3_morphometry", "label": "Tier 3: + Brain morphometry",
     "groups": ["demographics", "personality", "motivation_affect", "identity_belief",
                "brain_morphometry"]},
    {"id": "T4_multimodal_full", "label": "Tier 4: + Brain connectome (everything together)",
     "groups": ["demographics", "personality", "motivation_affect", "identity_belief",
                "brain_morphometry", "brain_connectome"]},
    {"id": "T5_brain_only", "label": "Tier 5: Brain only (morphometry + connectome)",
     "groups": ["brain_morphometry", "brain_connectome"]},
    {"id": "B1_morphometry_only", "label": "Brain-only probe: morphometry",
     "groups": ["brain_morphometry"]},
    {"id": "B2_connectome_only", "label": "Brain-only probe: connectome",
     "groups": ["brain_connectome"]},
]

ONTOLOGY_CONTEXT = (
    "AOMIC ID1000: 928 healthy Dutch young adults. The features grouped here are "
    "self-report / questionnaire measures: NEO-FFI Big Five personality, BIS/BAS "
    "reinforcement sensitivity, STAI trait anxiety, demographics, anthropometrics, "
    "socio-economic status, sexual/gender identity ratings and religiosity. Group "
    "measures from the same instrument or construct together. (Brain imaging "
    "features - FreeSurfer morphometry and the functional connectome - are placed "
    "deterministically under a single BRAIN domain via explicit paths and are not "
    "part of this semantic grouping step.)"
)

# ----------------------------------------------------------- subset run
# Deterministic, target-blind evaluation cohort for the live LLM run.
SUBSET_SIZE = 100
RANDOM_SEED = 42
MAX_ITERATIONS = 1  # actor-critic loop depth
EVALUATION_IDS_PATH = RESULTS_DIR / "evaluation_ids.json"

# Core columns required to be present for a subject to be eligible for the subset.
CORE_COMPLETE = ["NEO_N", "NEO_E", "NEO_O", "NEO_A", "NEO_C",
                 "BAS_drive", "BIS", "STAI_T", "BMI"]


def evaluation_candidate_ids(df):
    """Return initial and backup candidates in a deterministic target-blind order."""
    import numpy as np
    import pandas as pd
    target = TARGET["column"]
    ok = df[pd.to_numeric(df["BMI"], errors="coerce").fillna(0) > 0].copy()
    for col in CORE_COMPLETE:
        ok = ok[pd.to_numeric(ok[col], errors="coerce").notna()]
    ok = ok[pd.to_numeric(ok[target], errors="coerce").notna()]
    candidates = np.array(sorted(ok["participant_id"].astype(str).unique()))
    if len(candidates) < SUBSET_SIZE:
        raise ValueError(
            f"Requested {SUBSET_SIZE} evaluation participants but only "
            f"{len(candidates)} are eligible"
        )
    rng = np.random.default_rng(RANDOM_SEED)
    initial = sorted(rng.choice(candidates, size=SUBSET_SIZE, replace=False).tolist())
    initial_set = set(initial)
    backups = [pid for pid in rng.permutation(candidates).tolist() if pid not in initial_set]
    return initial + backups


def select_subset_ids(df, use_lock=True):
    """Select or load a deterministic evaluation cohort without target values.

    Eligibility uses predictor completeness and the presence, but never the value,
    of the target. The initial seeded draw may be backfilled only for unavailable
    imaging. Step 11 writes that final modality-complete set to a lock file.
    """
    import json
    candidates = evaluation_candidate_ids(df)
    eligible = set(candidates)
    if use_lock and EVALUATION_IDS_PATH.exists():
        locked = json.loads(EVALUATION_IDS_PATH.read_text()).get("participant_ids", [])
        if len(locked) != SUBSET_SIZE or len(set(locked)) != SUBSET_SIZE:
            raise ValueError(f"Invalid evaluation lock: {EVALUATION_IDS_PATH}")
        if not set(locked).issubset(eligible):
            raise ValueError("Evaluation lock contains ineligible participant IDs")
        return sorted(locked)
    return sorted(candidates[:SUBSET_SIZE])


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


def features_for_ontology(df=None):
    """Rich feature descriptors for semantic, LLM-driven ontology construction.

    Provides label, description, units, statistical type, measurement source, and a
    small sample of observed values so the model has everything it needs to group by
    meaning. No target domain/subdomain is imposed; the model proposes the hierarchy.
    """
    import pandas as pd
    specs = all_feature_specs()
    out = []
    for col, spec in specs.items():
        sample = None
        if df is not None and col in df.columns:
            vals = df[col].dropna().unique()[:4]
            sample = [str(v) for v in vals]
        feat = {
            "id": col,
            "label": spec.get("label", col),
            "definition": spec.get("description", ""),
            "stat_type": spec.get("stat_type", "numeric"),
            "units": spec.get("units"),
            "source": SOURCE_INFO.get(spec.get("group", ""), ""),
            "sample": sample,
        }
        # Brain modalities carry an explicit deep ontology path (deterministic,
        # high-resolution). Tabular features have none -> grouped semantically by
        # the LLM. Both end up in one arbitrary-depth ontology.
        if spec.get("path"):
            feat["path"] = spec["path"]
        out.append(feat)
    return out
