# INTELLIGENCE (AOMIC-ID1000, OpenNeuro ds003097): tiers and phenotype structure

Cohort: 928 healthy Dutch young adults (CC0). Evaluation cohort: a deterministic,
target-blind slice of 100 participants (`results/subset.json`), z-scored against a
disjoint reference split so a subject is never standardized against itself.

## Prediction target (clinical phenotype)

Native intelligence on the Intelligence Structure Test 2000-R (IST 2000-R). All
scores are native IST points (sums of correct items), NOT a conventional 100/15 IQ
scale. The phenotype is predicted as a two-level hierarchy:

```
total_intelligence            univariate regression   IST_intelligence_total  (native IST points)
  └─ ist_subscales            multivariate regression  IST_fluid, IST_memory, IST_crystallised
```

- Root node `total_intelligence`: one continuous value, the overall IST composite.
- Child node `ist_subscales`: three continuous values predicted jointly:
  - `IST_fluid`: reasoning on novel verbal, numerical and figural problems.
  - `IST_memory`: short-term retention and recall of verbal and figural material.
  - `IST_crystallised`: acquired verbal and numerical knowledge.

The three subscales are components of the total, so they tend to move together with
it. They are prediction OUTPUTS only; they are never predictors (all four IST
columns are in `config.EXCLUDED_COLUMNS`), so there is no target leakage. Ground
truth for all four outputs, per evaluation subject, is in `results/annotations.json`.

## Data-complexity tiers

Each tier is a cumulative bundle of feature groups projected onto the fixed master
ontology. A tier is usable only if every group it names has extracted features.

| Tier id | Adds | Modalities in the tier |
|---|---|---|
| `T1_demographics` | demographics | age, sex, handedness, BMI, education, background SES |
| `T2_personality_psychometric` | + personality and psychometrics (all self-report) | T1 + NEO-FFI Big Five, BIS/BAS reinforcement sensitivity, STAI trait anxiety |
| `T3_identity` | + identity and belief (all self-report) | T2 + sexual/gender identity ratings, religiosity |
| `T4_morphometry` | + brain structure | T3 + FreeSurfer cortical/subcortical morphometry |
| `T5_connectome` | + brain function (full multimodal) | T4 + movie-fMRI functional connectome (Yeo networks) |

The two self-report questionnaire tiers (NEO Big Five and the BIS/BAS + STAI psychometrics) are merged
into one, since both are self-report; the two brain tiers stay separate (structural vs functional).

Brain-only probes (no self-report) are also built: `B1_morphometry_only`,
`B2_connectome_only`, `B3_brain_only`. The cumulative ladder T1 to T6 is the primary
data-complexity axis: the question is how far up the modality ladder the engine must
climb before it recovers the intelligence phenotype.

## How the engine reads it

The global instruction injects the IST context (what the instrument is, that scores
are native points, that subscales are components of the total) plus reference-split
mean/sd per output, so the model predicts on the correct native scale. No IST value
for the evaluation subject is ever provided; every score is inferred from the
non-cognitive multimodal evidence in the tier.
