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
| `T4_eeg_lean` | 29 psychosis-implicated EEG features only | neural floor (targeted) |
| `T5_eeg_rich` | all 836 resting-EEG features only | neural ceiling (unguided) |

The lean EEG set (T4) is the psychosis-implicated subset: posterior alpha deficit,
frontal/global slow-wave excess, slowing ratios, and alpha-peak slowing. T4 vs T5
asks whether a small, theory-driven neural bundle matches the full 836-feature set;
T2 vs T3 asks what the EEG adds on top of the clinical profile.

## Ontology note (abstract structure)

All 935 predictors carry explicit ontology paths, so the hierarchy is deterministic
and deep (max depth 5). The 836 EEG features nest under Resting-State EEG by family
(spectral, alpha peak, aperiodic, entropy, fractal, microstates, connectivity,
graph). The Social Functioning Scale is broken out of a single flat 74-item list
into the instrument's own question blocks (withdrawal and engagement, interpersonal
and communication, activities and recreation, independence and competence,
employment), so the engine can prioritise and attribute evidence by named branch.
