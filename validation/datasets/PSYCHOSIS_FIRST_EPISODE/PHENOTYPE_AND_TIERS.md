# PSYCHOSIS_FIRST_EPISODE (OpenNeuro ds003944 + ds003947): tiers and phenotype structure

Cohort: 143 resting-EEG recordings on one harmonized 49-channel montage, 81
first-episode-psychosis cases and 62 controls (134 with complete EEG). Predictor
z-scores are referenced to a disjoint reference split; the balanced evaluation
cohort is 5 psychosis (spread across BPRS severity) plus 5 controls. Ground truth
for every recording is in `results/compass/annotations.json`.

## Prediction target (clinical phenotype)

A mixed-type, three-level hierarchy: a diagnosis at the root, overall symptom
severity beneath it, then the positive- and negative-symptom profiles beneath that.

```
diagnosis                     binary classification    Control vs First-Episode Psychosis
  └─ global_severity          univariate regression    BPRS 19-item total (each item 1-7; total 19-133)
       ├─ positive_symptoms   multivariate regression  4 SAPS global ratings (0-5 each)
       └─ negative_symptoms   multivariate regression  5 SANS global ratings (0-5 each)
```

- `diagnosis`: case vs control, with class probabilities.
- `global_severity`: Brief Psychiatric Rating Scale total, the overall severity scalar.
- `positive_symptoms` (SAPS globals): hallucinations, delusions, bizarre behavior,
  positive formal thought disorder.
- `negative_symptoms` (SANS globals): affective flattening, alogia, avolition/apathy,
  anhedonia/asociality, attention.

Controls are expected to have low or absent symptoms; cases vary widely. Every
BPRS/SAPS/SANS item is a target and is psychosis-only, so none is ever a predictor.

## Leakage control

Never predictors: the diagnosis label, every BPRS/SANS/SAPS item, the GAS/global
assessment and SFS employment items (control-sparse, would leak the label through
missingness), and chlorpromazine-equivalent dose (psychosis-only). See
`compass_task.excluded_columns`.

## Data-complexity tiers

| Tier id | Content | Intent |
|---|---|---|
| `T1_demographic_socioeconomic` | demographics + Hollingshead socio-economic status | non-clinical floor |
| `T2_clinical_profile` | + MATRICS cognition, WASI IQ, GAF and SFS observed functioning | full non-neural clinical profile |
| `T3_multimodal_full` | + all 836 resting-EEG features | full multimodal ceiling |
| `T4_eeg_brain_only` | 79-feature psychosis-signature resting EEG only | brain-only (curated) |

The brain-only tier (T4) is a curated 79-feature psychosis signature across six well-replicated
families: posterior alpha deficit and frontal/global slow-wave excess with slowing ratios (spectral),
alpha-peak slowing, aperiodic 1/f exponent/offset (excitation-inhibition balance), reduced signal
complexity (Lempel-Ziv, sample and permutation entropy), microstate C excess and D deficit with C-D
transitions, and compact alpha-band connectivity/graph summaries. It replaces the former two EEG-only
tiers (a 29-feature lean and the full 836 rich): the full 836-only tier was about 95% redundant with the
multimodal ceiling T3, so it was dropped, and the lean was enriched into this signature. T2 vs T3 asks
what the EEG adds on top of the clinical profile; T4 asks what the neural signature carries on its own.

## Ontology note (abstract structure)

All 935 predictors carry explicit ontology paths, so the hierarchy is deterministic
and deep (max depth 5). It has three primary domains:

1. **Resting EEG** (836 leaves): the EEG features nest by family (spectral, alpha
   peak, aperiodic, entropy, fractal, microstates, connectivity, graph).
2. **Demographics and Socio-economic Status** (8 leaves): participant demographics
   and Hollingshead socio-economic status.
3. **Global Functioning** (91 leaves): the psychiatric superordinate construct,
   gathering at one secondary level both the cognitive/intelligence measures (MATRICS
   cognitive domains, WASI IQ) and the observed/social functioning measures (Global
   Assessment of Functioning ratings, Social Functioning Scale). This merges what
   were previously two separate domains (Cognition and Intelligence, Observed
   Functioning) into one, so the two social nodes now sit alongside the two
   intelligence nodes.

The Social Functioning Scale is broken out of a single flat 74-item list into the
instrument's own question blocks (withdrawal and engagement, interpersonal and
communication, activities and recreation, independence and competence, employment),
so the engine can prioritise and attribute evidence by named branch.
