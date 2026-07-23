# NUMERACY_STROKE (OpenNeuro ds006533): tiers and phenotype structure

Cohort: 105 left-hemisphere chronic stroke survivors assessed on two numeracy
systems. A seeded, target-blind 20-subject slice per target is the leakage-safe
blinded evaluation cohort (`results/subset_<target>.json`), z-scored against the
disjoint reference; the full 105-subject ground truth for both targets is in
`results/annotations.json`.

## Prediction target (clinical phenotype)

Two dissociable numeracy phenotypes, each predicted as its own univariate regression
on its native population Z-score scale (0 = stroke-cohort mean, +/-1 = one SD, higher
is better):

```
approximate_numeracy   univariate regression   non-symbolic Approximate Number System (dot comparison)
precise_numeracy       univariate regression   precise symbolic numeracy (WAB number items, writing, dictation, calculation)
```

They are kept as SEPARATE univariate tasks rather than one joint target because the
dataset's scientific point is their DIFFERENTIAL relationship to language and lesion
features: approximate numeracy is largely spared by left-hemisphere language damage,
whereas precise, symbolic numeracy is more vulnerable to aphasia and peri-sylvian /
parietal lesions. Running both across the tier ladder and comparing per-tier
recovery is the readout. The two FA factor scores are excluded as predictors (they
are derived from the same task battery as the targets, so circular), and
`aphasia_quotient` stays a predictor.

## Data-complexity tiers (transformed tier only)

Only the TRANSFORMED feature tier is ingested. The two scalar clinical covariates are
already standardized on disk (their units are labelled as standardized transforms,
not their raw native scale); the per-region lesion-overlap features are raw 0-1
proportions. The engine is told this explicitly in a SCALE GUIDE inside each
participant's `non_numerical_data.txt` and in the run-time global instruction.

| Tier level | Adds | Features |
|---|---|---|
| `T1_demographics` | demographics | age, education years, imaging modality |
| `T2_aphasia` | + clinical severity | T1 + standardized WAB-R aphasia quotient, standardized whole-brain lesion volume |
| `T3_lesion_fine` | + per-parcel lesion load (everything together) | T2 + prevalence-filtered per-ROI lesion overlap (194 features) |
| `T4_lesion_brain_only` | brain only: lesion topography | per-parcel lesion overlap only (189 features), no demographics or aphasia |

The ladder builds to everything together (T3), then a brain-only lesion tier (T4) isolates the anatomical
contribution (per-parcel lesion overlap only, no demographics or clinical severity). The coarse
network-level lesion tier was dropped as redundant with the fine per-parcel map (the fine parcels
aggregate to those networks).

## Ontology note (abstract structure)

The flat lesion table is organised into a deterministic, deep abstract hierarchy
(children schema, max depth 3-4): `Demographics and Background`, `Clinical Profile`
(language function; whole-brain lesion load), and `Brain Lesion Topography` resolved
as Cerebral cortex (Schaefer 2018 / Yeo 7 networks) -> network -> parcel; Subcortical
nuclei (Tian S4) -> structure -> parcel; Cerebellum (Nettekoven 2024) -> functional
domain -> subregion -> parcel. This lets the engine reason about lesion load at the
network level and drill into individual parcels within the same tree.
