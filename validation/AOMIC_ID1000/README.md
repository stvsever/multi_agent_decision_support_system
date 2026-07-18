# AOMIC-ID1000 (OpenNeuro ds003097)

Validation of the COMPASS engine on a real, fully open-access dataset that has a
completely different structure from the UK Biobank layout the engine was
originally shaped around. This folder is self-contained: the dataset, the
preprocessing pipeline, the generated ontology, the engine inputs, and all
results live here. Nothing dataset-specific was added to `src/full_stack`.

Task: **univariate regression of total intelligence** from non-cognitive
phenotype features only.

---

## 1. Dataset

| | |
|---|---|
| Name | AOMIC-ID1000 |
| Source | OpenNeuro `ds003097` |
| DOI | `10.18112/openneuro.ds003097.v1.2.1` |
| License | CC0 (public domain) |
| Participants | 928 healthy Dutch young adults (ages 19-26) |
| Modality used here | Tabular phenotype (`participants.tsv` + `participants.json` data dictionary) |

The Amsterdam Open MRI Collection ID1000 study collected self-report
questionnaires, demographics, and an intelligence test alongside MRI. This
validation uses only the pre-processed tabular phenotype, which is enough to
exercise the full ingestion, ontology, and multi-agent regression flow.

Raw files are stored verbatim under [`dataset/`](dataset/):
`participants.tsv`, `participants.json` (per-column descriptions), and
`dataset_description.json`.

## 2. Prediction task

Predict **`IST_intelligence_total`** (the Intelligence Structure Test 2000-R
composite; cohort mean ~200, sd ~40, range 68-296) as a single number.

Excluded from the predictors (see [`pipeline/config.py`](pipeline/config.py)):

- `IST_fluid`, `IST_memory`, `IST_crystallised` - the target's own subscales,
  so using them would be circular.
- `IST_intelligence_total` - the target itself.
- `DWI_TR_run1/2/3` - MRI acquisition parameters, not phenotypic.

That leaves **23 genuine predictors** spanning personality (NEO-FFI Big Five),
reinforcement sensitivity (BIS/BAS), trait anxiety (STAI-T), demographics,
anthropometrics, socio-economic status, sexual/gender identity, and religiosity.
Predicting intelligence from these non-cognitive features is intentionally hard;
the goal is to validate the *system*, not to maximise accuracy.

Only the target's scale (mean/sd/range, not any participant value) is passed to
the engine as a global calibration instruction. This is measurement metadata,
not label leakage.

## 3. Pipeline (clean, sequential)

Each step is a standalone script in [`pipeline/`](pipeline/). Dataset-agnostic
logic it relies on lives in [`../common/`](../common/); only
[`pipeline/config.py`](pipeline/config.py) holds AOMIC-specific knowledge.

| Step | Script | LLM? | What it does |
|---|---|---|---|
| 01 | `01_explore_structure.py` | no | Hardcoded profiling of every column (type, missingness, distribution). Writes `ontology/feature_manifest.json`. |
| 02 | `02_build_ontology.py` | yes | LLM builds a non-redundant DOMAIN -> SUBDOMAIN -> FEATURE subclass ontology; validated and repaired for full coverage. Writes `subclass_structure.json` and `aomic_id1000.owl`. |
| 03 | `03_build_compass_inputs.py` | optional | Fits the reference model, selects a subset, and projects each participant onto the ontology as the four COMPASS files. |
| 04 | `04_run_compass.py` | yes | Runs the real engine pipeline (`run_compass_pipeline`) for each subset participant via OpenRouter. |
| 05 | `05_evaluate.py` | no | Computes error and rank-agreement metrics. |

Run it:

```bash
cd validation/AOMIC_ID1000/pipeline
python3 run_all.py          # steps 01-03 (offline except the single ontology call)
python3 run_all.py --live   # also runs 04 (engine) + 05 (evaluation)
```

The engine backend, model, and OpenRouter key come from the repository-root
`.env` (`OPENROUTER_API_KEY`). All roles use `google/gemini-3.1-flash-lite`.

## 4. LLM-based ontology

Step 02 asks a small model to organise all 23 features into a strict,
non-redundant IS-A hierarchy. The proposal is then validated in code: every
feature must appear as a leaf exactly once, and any duplicate/unknown/forgotten
feature is repaired automatically before serialisation.

Result for this dataset (3 domains, 7 subdomains, 23 leaves, zero repairs
needed):

```
DEMOGRAPHICS_AND_PHYSICAL
  personal_attributes      : age, sex, handedness, BMI
  socioeconomic_status     : education_level, background_SES
PSYCHOMETRIC_PROFILES
  big_five_personality     : NEO_N, NEO_E, NEO_O, NEO_A, NEO_C
  reinforcement_sensitivity: BAS_drive, BAS_fun, BAS_reward, BIS
  anxiety_traits           : STAI_T
IDENTITY_AND_BELIEF
  sexual_and_gender_identity: sexual_attraction_M/F, gender_identity_M/F
  religiosity              : religious_upbringing, religious_now, religious_importance
```

Two artifacts are written to [`ontology/`](ontology/):

- **`aomic_id1000.owl`** - RDF/XML OWL with `rdfs:subClassOf` hierarchy, labels,
  and definitions. Loads directly in **Protege** (Open -> select the file) for
  visual inspection of the class tree.
- **`subclass_structure.json`** - the same tree machine-readable, plus a
  `column_index` mapping each source column to its ontology path. This is what
  step 03 consumes to build the engine inputs.

## 5. Flexible ingestion and reference strategies

The engine expects a "UKB nested" contract (four files per participant). Step 03
projects the flat AOMIC table onto that contract, so the engine ingests it with
no code changes. Feature **labels** are preserved next to every value (for
example `Openness (NEO-FFI) = 51 (High-Normal)`) because the multi-agent system
reasons over text, not bare numbers.

Not every dataset has a normative reference, so the reference strategy is
explicit and auto-selected (`validation/common/deviation.py`):

- **cohort** - standardise against this batch's own mean/sd (used here, n=928).
- **external** - standardise against supplied normative stats.
- **absolute** - no reference available (single-subject inference or no reference
  group): raw pre-processed values are passed through, and an optional one-shot
  LLM call estimates plausible ranges so the engine still sees High/Normal/Low
  context.

Set `REFERENCE_MODE` in `config.py` (default `auto`). Generated files per
participant, under [`compass_inputs/`](compass_inputs/):
`data_overview.json`, `hierarchical_deviation_map.json`, `multimodal_data.json`,
`non_numerical_data.txt`.

## 6. Results (subset smoke run)

A deterministic subset of **6 participants** spanning the full IQ range (68-296)
was run through the real actor-critic pipeline with `google/gemini-3.1-flash-lite`
(one orchestration iteration each). All six produced a valid numeric regression
output and were rated SATISFACTORY by the critic; each run took ~25 s.

| Participant | Ground truth | Predicted | Abs error |
|---|---|---|---|
| sub-0120 | 296 | 217.5 | 78.5 |
| sub-0128 | 235 | 235.0 | 0.0 |
| sub-0153 | 193 | 210.8 | 17.8 |
| sub-0282 | 68 | 185.5 | 117.5 |
| sub-0498 | 166 | 188.0 | 22.0 |
| sub-0673 | 216 | 192.0 | 24.0 |

**Metrics** (`results/metrics.json`):

| Metric | Value |
|---|---|
| MAE | 43.3 points |
| RMSE | 59.7 points |
| Normalised MAE (vs target sd) | 0.62 |
| R2 | 0.27 |
| Pearson r | 0.69 |
| Spearman rho | 0.89 |
| Rank stability (bootstrap Spearman) | 0.85 +/- 0.18 |
| Leave-one-out Spearman (min / mean) | 0.80 / 0.87 |

Reading: point error is large (expected when inferring intelligence from
personality and demographics with a tiny model), but the **rank recovery is
strong and stable** - the engine ordered these six participants by intelligence
almost correctly (Spearman 0.89) from non-cognitive evidence alone. The single
large miss is the cohort-minimum participant (68), which regresses toward the
mean. Rank stability quantifies how robust that ordering is to resampling; a high
value with a small subset is a good sign for the ingestion and reasoning flow.

N=6 metrics are indicative only. To scale up, raise `SUBSET_SIZE` and
`MAX_ITERATIONS` in `config.py`.

## 7. Folder layout

```
AOMIC_ID1000/
  README.md                     this file
  dataset/                      raw CC0 data + data dictionary
  pipeline/                     config + numbered sequential scripts
  ontology/                     feature_manifest.json, subclass_structure.json, aomic_id1000.owl
  compass_inputs/               generated engine inputs (4 files x 6 participants)
  results/
    subset.json                 subset + ground truth
    predictions.json            predicted vs ground truth per participant
    metrics.json                aggregate metrics
    compass_runs/               full per-participant engine outputs (reports, logs)
```

## 8. Reproducing / adding a dataset

To validate another dataset, copy this folder, replace `dataset/`, and edit
`pipeline/config.py` (target, excluded columns, feature specs, reference mode).
The numbered scripts and `../common/` helpers are reused unchanged.
