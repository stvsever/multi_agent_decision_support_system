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

## 4. Automated exploration and the LLM-based master ontology

Ontology construction is preceded by a sophisticated, dataset-agnostic **automated
exploration** ([`common/explore.py`](../../common/explore.py)): automatic type
inference, distribution and missingness profiling, robust (Spearman) correlation
structure, hierarchical feature clustering, near-duplicate detection, and target
associations. On this dataset it flags real redundancies (total-gray vs cortical
volume r=0.97, left vs right mean thickness r=0.96, sexual-attraction M/F r=-0.95) and
proposes data-driven clusters, written to `ontology/exploration_report.json`.

One non-redundant `DOMAIN -> SUBDOMAIN -> FEATURE` ontology is then built over all 86
features. The structure is fixed from per-feature hints (guaranteeing complete,
non-redundant coverage and keeping brain morphometry and connectome as separate
domains); a small model generates only the interpretable parent labels and
definitions. Leaf features store a self-explanatory label and no redundant
description; only parent nodes carry LLM-written definitions. The prompt is serialised
as TOON to keep it token-cheap even at 86 features.

The ontology is then **quality-assessed** (`ontology/ontology_report.json`): the
adjusted Rand index between the semantic subdomains and the data-driven clusters
(~0.48 here, i.e. meaning-based grouping that partly but not slavishly follows
statistics), plus a compact LLM review for MECE / non-redundancy / coherence. Result:

```
DEMOGRAPHICS_AND_PHYSICAL   personal_attributes, anthropometrics, socioeconomic_status
PERSONALITY                 big_five_personality
MOTIVATION_AND_AFFECT       reinforcement_sensitivity, anxiety_traits
IDENTITY_AND_BELIEF         sexual_and_gender_identity, religiosity
BRAIN_MORPHOMETRY           global_brain_measures, subcortical_volumes, cortical_thickness
BRAIN_CONNECTOME            within_network_connectivity, between_network_connectivity
```

Artifacts in [`ontology/`](ontology/): `aomic_id1000.owl` (loads in Protege),
`subclass_structure.json`, `feature_manifest.json`, `exploration_report.json`, and
`ontology_report.json`. The ontology is built once and reused as a fixed base template
for every subject and every tier, so there is no per-individual structural variation,
only differences in which leaf values are present.

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
N=6 the point error is noisy; **rank recovery** (Spearman) and its bootstrap stability
are the informative signals.

| Tier | Features | MAE | nMAE | Pearson r | Spearman rho | Rank stability |
|---|---|---|---|---|---|---|
| T1 Demographics | 6 | 43.2 | 1.07 | 0.78 | 0.71 | 0.65 |
| T2 + Personality | 11 | 44.3 | 1.10 | 0.85 | 1.00 | 1.00 |
| T3 + Motivation & Affect | 16 | 41.4 | 1.03 | 0.87 | 0.71 | 0.70 |
| T4 + Identity & Belief | 23 | 44.9 | 1.11 | 0.85 | 0.54 | 0.55 |
| T5 + Brain morphometry | 58 | 45.5 | 1.13 | **0.91** | 0.83 | 0.81 |
| T6 + Brain connectome (full) | 86 | **40.6** | **1.00** | 0.78 | 0.60 | 0.59 |
| B1 Brain-only morphometry | 35 | 52.4 | 1.30 | 0.55 | 0.77 | 0.73 |
| B2 Brain-only connectome | 28 | 42.3 | 1.05 | 0.87 | 0.66 | 0.67 |
| B3 Brain-only | 63 | 47.3 | 1.17 | 0.76 | 0.37 | 0.43 |

Reading: adding personality (T2) and morphometry (T5) give the strongest linear
tracking of intelligence (Pearson up to 0.91), and the full multimodal tier (T6) gives
the best absolute error (nMAE 1.00). The brain-only connectome tier alone reaches
Pearson 0.87, consistent with naturalistic-viewing FC carrying individual-difference
signal. These are 6-subject numbers and fluctuate; they validate the multi-modal
ingestion, ontology, and end-to-end engine flow rather than establishing predictive
performance. Raise `SUBSET_SIZE` / `MAX_ITERATIONS` in `config.py` to scale up.

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
