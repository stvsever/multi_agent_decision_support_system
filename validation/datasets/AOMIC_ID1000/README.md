# AOMIC-ID1000 (OpenNeuro ds003097)

Validation of the COMPASS engine on a real, fully open-access dataset with a
structure unlike the UK Biobank layout the engine was originally shaped around. This
folder is self-contained: raw data, preprocessing pipelines, the generated ontology,
per-tier engine inputs, results, and visualisation notebooks all live here. Nothing
dataset-specific was added to `src/full_stack`.

Task: **univariate regression of total intelligence**, evaluated across a ladder of
data-complexity tiers that add one modality at a time, up to a full multi-modal
input, plus brain-only tiers.

---

## 1. Dataset

| | |
|---|---|
| Name | AOMIC-ID1000 |
| Source | OpenNeuro `ds003097` (DOI `10.18112/openneuro.ds003097.v1.2.1`) |
| License | CC0 (public domain) |
| Participants | 928 healthy Dutch young adults (ages 19-26) |
| Modalities used | Tabular self-report, FreeSurfer morphometry, movie-fMRI connectome |

Target: **`IST_intelligence_total`** (Intelligence Structure Test 2000-R composite;
cohort mean ~200, sd ~40, range 68-296). The three IST subscales and the composite
itself are excluded from predictors (circular); MRI scan parameters are excluded as
non-phenotypic. Only the target's scale (mean/sd, not any participant value) is passed
to the engine as calibration.

## 2. The three modalities

**Self-report (tabular).** 23 features straight from `participants.tsv`: NEO-FFI Big
Five personality, BIS/BAS, STAI trait anxiety, demographics, anthropometrics, SES,
sexual/gender identity, religiosity.

**Brain morphometry (FreeSurfer).** 35 features parsed from the per-subject FreeSurfer
stats tables and reduced to a clean, labeled set: global measures (intracranial and
gray/white volumes), subcortical volumes for key bilateral structures, and cortical
thickness aggregated to lobes (surface-area weighted). See
[`common/freesurfer.py`](../../common/freesurfer.py) and step `10`.

**Brain connectome (movie-watching fMRI).** Raw voxel connectivity has far too many
edges to hand to the engine, so this is a real dimensionality-reduction pipeline:
fMRIPrep BOLD is parcellated with Schaefer-100 / Yeo-7, confound-denoised, and reduced
to 28 **network-level** functional connectivity features (7 within-network, 21
between-network). See [`common/connectome.py`](../../common/connectome.py) and step `11`.

## 3. Complexity tiers

Each cumulative tier adds a modality block; brain-only tiers isolate the imaging. Every
tier is a filtered projection of the single master ontology, so they stay mutually
consistent.

| Tier | Adds | Feature groups |
|---|---|---|
| T1 Demographics | baseline | demographics |
| T2 + Personality | Big Five | + personality |
| T3 + Motivation & Affect | BIS/BAS, anxiety | + motivation_affect |
| T4 + Identity & Belief | all self-report | + identity_belief |
| T5 + Brain morphometry | structure | + brain_morphometry |
| T6 + Brain connectome | full multimodal | + brain_connectome |
| B1 Brain-only morphometry | isolate structure | brain_morphometry |
| B2 Brain-only connectome | isolate function | brain_connectome |
| B3 Brain-only | structure + function | brain morphometry + connectome |

## 4. Semantic, LLM-driven ontology construction

Construction is preceded by a dataset-agnostic **automated exploration**
([`common/explore.py`](../../common/explore.py)): type inference, distribution and
missingness profiling, robust (Spearman) correlation structure, feature clustering,
near-duplicate detection, and target associations, written to
`ontology/exploration_report.json`. This exploration documents the data and later
quality-checks the ontology, but it deliberately **does not drive the structure**.
Grouping features by how they correlate would be wrong: two semantically unrelated
measures (say a subcortical volume and a verbal-reasoning score) can correlate by
accident. The ontology must group by **meaning**.

So the ontology is built by the LLM, semantically, and it is general to any dataset
([`common/ontology.py`](../../common/ontology.py), `build_semantic_ontology`):

1. the model is given every feature with its label, description, units, measurement
   source, and a few sample values (serialised as TOON for token efficiency),
2. it **proposes the domains itself** (not hardcoded), grouping by meaning and source,
3. for each proposed domain in parallel, it selects that domain's features and
   organises them into subdomains,
4. code then enforces exact coverage and non-redundancy (a feature claimed twice is
   kept once; anything unclaimed is repaired into an explicit `UNASSIGNED` domain).

Decomposing per domain keeps every prompt small, so it scales to large multimodal
feature sets ("commands" rather than one giant call). An optional **free-text user
guidance** argument is injected into every prompt; this is the backend hook for a
future UI where a user can steer the ontology in natural language
(`--guidance "..."` on step 02, or `ONTOLOGY_USER_GUIDANCE` in config).

On this dataset the model proposed 5 domains over 86 features (it chose to fold
personality, motivation and anxiety into one "Psychological Profiles" domain, and kept
morphometry and connectome separate):

```
DEMOGRAPHICS_AND_ANTHROPOMETRICS   biological_characteristics, socioeconomic_background
PSYCHOLOGICAL_PROFILES             personality_traits, reinforcement_sensitivity, trait_anxiety
IDENTITY_AND_BELIEFS               sexual_and_gender_identity, religious_affiliation_and_importance
BRAIN_MORPHOMETRY                  global_volumetric_measures, subcortical_volumes, cortical_thickness
FUNCTIONAL_CONNECTOME              within_network_connectivity, between_network_connectivity
```

The ontology is **quality-assessed** (`ontology/ontology_report.json`): the adjusted
Rand index against the purely statistical clusters (~0.48, confirming the ontology is
meaning-based rather than statistics-driven) plus a compact LLM MECE/coherence review.

Artifacts in [`ontology/`](ontology/):

- `aomic_id1000.owl` - loads in Protege,
- `subclass_structure.json` - the machine-readable tree and column index,
- `ontology_features.csv` - a single flat benchmark matrix: 928 participants x all 86
  features, with the hierarchy encoded in each column name
  (`DOMAIN|subdomain|feature`) plus the target, ready for ML comparison,
- `ontology_viewer.html` - a self-contained interactive explorer (expand/collapse,
  drag, and top-down / left-right / radial layouts),
- `exploration_report.json`, `ontology_report.json`, `feature_manifest.json`.

The ontology is built once and reused as a fixed base template for every subject and
every tier, so there is no per-individual structural variation, only differences in
which leaf values are present.

## 5. Pipeline

Scripts in [`pipeline/`](pipeline/); dataset-agnostic logic is in
[`../../common/`](../../common/); AOMIC knowledge is only in
[`pipeline/config.py`](pipeline/config.py).

| Step | Script | What it does |
|---|---|---|
| 10 | `10_extract_freesurfer.py` | Download + parse FreeSurfer stats -> morphometry features |
| 11 | `11_extract_connectome.py` | Download fMRIPrep BOLD -> Schaefer/Yeo network FC features |
| 01 | `01_explore_structure.py` | Profile all features -> `feature_manifest.json` |
| 02 | `02_build_ontology.py` | Build the master ontology (one LLM labeling call) |
| 03 | `03_build_compass_inputs.py` | Project ontology per tier -> COMPASS inputs |
| 04 | `04_run_compass.py` | Run the engine per tier via OpenRouter |
| 05 | `05_evaluate.py` | Per-tier metrics + `tiers_summary.json` |

```bash
cd validation/datasets/AOMIC_ID1000/pipeline
python3 run_tiers.py            # offline: extraction, ontology, tier inputs
python3 run_tiers.py --live     # also run the engine on every tier + evaluate
python3 run_tiers.py --live --skip-brain   # reuse already-extracted brain features
```

## 6. Results (subset of 6 participants, `google/gemini-3.1-flash-lite`, 1 iteration)

All tiers produced valid numeric regressions rated SATISFACTORY by the critic. With
N=6 every number is noisy, so read trends across tiers, not any single cell. R2 is
reported alongside rank recovery (Spearman) and its bootstrap stability.

| Tier | Features | R2 | MAE | RMSE | Pearson r | Spearman rho | Rank stability |
|---|---|---|---|---|---|---|---|
| T1 Demographics | 6 | 0.10 | 53.8 | 70.2 | 0.56 | 0.50 | 0.51 |
| T2 + Personality | 11 | 0.15 | 49.5 | 64.2 | 0.63 | 0.49 | 0.49 |
| T3 + Motivation & Affect | 16 | 0.22 | 42.4 | 61.7 | 0.67 | 0.94 | 0.89 |
| T4 + Identity & Belief | 23 | 0.05 | 48.8 | 67.9 | 0.37 | 0.49 | 0.47 |
| T5 + Brain morphometry | 58 | 0.02 | 51.0 | 69.1 | 0.63 | 0.77 | 0.73 |
| T6 + Brain connectome (full) | 86 | **0.44** | 49.0 | **56.9** | **0.87** | **0.90** | **0.88** |
| B1 Brain-only morphometry | 35 | -0.03 | 52.4 | 70.8 | 0.20 | 0.03 | 0.12 |
| B2 Brain-only connectome | 28 | 0.21 | 45.7 | 62.0 | 0.65 | 0.60 | 0.59 |
| B3 Brain-only | 63 | 0.17 | 51.3 | 63.6 | 0.68 | 0.49 | 0.47 |

Reading, in trend terms:

- The **full multimodal tier (T6)** is clearly the strongest (R2 0.44, Pearson 0.87,
  Spearman 0.90, lowest RMSE), so combining self-report and brain modalities helps.
- Adding the **connectome** (T5 to T6) is the single biggest jump (R2 0.02 to 0.44).
- **Connectome-only (B2, R2 0.21)** beats **morphometry-only (B1, R2 about 0)** by a
  wide margin, consistent with naturalistic-viewing functional connectivity carrying
  more individual-difference signal for intelligence than regional structure at this
  sample size.
- The self-report ladder is non-monotonic at N=6 (T3 spikes, T4 dips), which is the
  expected small-sample noise.

These 5-to-6 subject numbers validate the multi-modal ingestion, semantic ontology,
tiering, and end-to-end engine flow, not predictive performance. Raise `SUBSET_SIZE`
and `MAX_ITERATIONS` in `config.py` to scale up.

Per-tier detail: `results/<tier>/predictions.json`, `results/<tier>/metrics.json`,
full engine outputs under `results/<tier>/compass_runs/`, and the cross-tier
`results/tiers_summary.json`.

## 7. Notebooks

Three executed, self-contained notebooks with ~26 embedded visualisations in
[`notebooks/`](notebooks/):

- `01_tabular_data_exploration.ipynb` - target and subscales, coverage and missingness
  co-occurrence, a distribution grid over every feature, clustered correlation heatmap
  and dendrogram, IQ associations, violins by education/sex, a pairplot, and a PCA of
  participants coloured by IQ.
- `02_brain_preprocessing.ipynb` - subcortical volume distributions and hemispheric
  symmetry, head-size scaling, a lobe-thickness map, morphometry correlations and IQ
  links, morphometry PCA, the Yeo-7 group and per-subject FC matrices (nilearn),
  within/between-network edge strengths, and a parcel-level FC matrix plus glass-brain
  connectome for one subject.
- `03_ontology_and_results.ipynb` - the ontology as a graph, features per domain,
  data-driven clusters and their agreement with the ontology, detected redundancies,
  the tier ladder, per-tier performance bars, rank stability, and one participant's
  deviation profile across all domains.

Regenerate with `python3 notebooks/build_notebooks.py`.

## 8. Reproducing / adding a dataset

Copy this folder under `validation/datasets/`, replace `dataset/`, and edit
`pipeline/config.py` (target, excluded columns, feature specs and groups, tiers,
reference mode). The numbered scripts and `../../common/` helpers are reused unchanged.
