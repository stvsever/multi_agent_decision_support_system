# AOMIC-ID1000 IQ inference validation methodology

## Purpose and prediction structure

This validation asks COMPASS to predict one continuous outcome for each held-out
participant: `IST_intelligence_total`, the native composite score from the
Intelligence Structure Test 2000-R. The engine performs zero-shot, participant-level
regression from non-cognitive evidence. It is not fitted on the evaluation labels and
it is not a conventional supervised model trained on this dataset.

For each participant and tier, the engine receives four files:

1. `data_overview.json` describes domain coverage and token counts.
2. `multimodal_data.json` contains ontology-organized feature leaves, raw labels,
   reference-split z scores, and qualitative deviation labels.
3. `hierarchical_deviation_map.json` summarizes signed deviations by subdomain.
4. `non_numerical_data.txt` gives the same evidence as a deterministic narrative and
   states the target scale.

The multi-agent sequence is:

```text
participant input
  -> orchestrator builds an evidence plan
  -> executor runs independent evidence tools in parallel
  -> integrator combines tool evidence
  -> predictor emits one native IST total score
  -> critic checks schema, grounding, and internal consistency
```

`MAX_ITERATIONS=1` is used for this benchmark. The point of the experiment is to
measure the inference system as configured, not to tune predictions against labels.

## Target scale and interpretable MAE

`IST_intelligence_total` is not reported by this dataset on the familiar IQ scale
with mean 100 and SD 15. Its native values are around mean 200 and SD 40. Therefore,
every result reports three forms of error:

- native IST MAE, in the actual units predicted by the engine;
- MAE divided by the disjoint reference-split target SD;
- an IQ-scale-equivalent MAE, computed as `15 * native_MAE / reference_SD`.

For participant-level display only, the corresponding linear transformation is:

```text
IQ-equivalent = 100 + 15 * (native_IST - reference_mean) / reference_SD
```

This is an interpretive cohort standardization. It is not an official IST norm
conversion and must not be described as a clinically normed IQ score. The exact
reference mean and SD used by a run are stored in `results/subset.json` and
`results/tiers_summary.json`.

## Participants, split, and blinding

The source table contains 928 participants. Evaluation eligibility requires a
non-missing target, valid BMI, and complete core personality, motivation, and affect
predictors. One hundred evaluation participants are drawn by a seeded random sample
from sorted eligible participant IDs with `RANDOM_SEED=42`.

Target values are never used to order, stratify, or select participants. Filtering
on whether an evaluation label exists is allowed; filtering on its numeric value is
not. The previous six-person range-spanning challenge sample is no longer used for
headline performance because selecting target extremes distorted R2 and rank metrics.

All normalization and target-scale calibration are fitted on participants outside the
100-person evaluation set. For brain-inclusive tiers, at least 20 non-evaluation
participants must have every brain feature. Step 03 stops instead of silently using
test-inclusive or undersized reference statistics.

OpenNeuro source participant IDs are replaced with `eval-0001` through `eval-0100` in
every model-visible file and directory. The private evaluation mapping remains in
`results/subset.json` for reproducibility, but it is never loaded into a prompt. This
blocks participant-ID lookup or memorization as an inference shortcut.

## Predictor inventory and semantic ontology

The final feature space contains 87 non-cognitive predictors:

| Block | Count | Contents |
|---|---:|---|
| Demographics and physical | 6 | age, sex, handedness, BMI, education, background SES |
| Personality | 5 | NEO-FFI Big Five scales |
| Motivation and affect | 5 | BIS, three BAS scales, trait anxiety |
| Identity and belief | 7 | attraction, gender identity, and religiosity items |
| Brain morphometry | 36 | 6 global volumes, 16 bilateral subcortical volumes, 14 cortical thickness summaries |
| Functional connectome | 28 | 7 within-network and 21 between-network Yeo-7 connectivity values |

The 36 morphometry features are a deliberate interpretable reduction of the much
larger raw FreeSurfer output. They are not all possible cortical measurements. The
pipeline retains global brain measures, key bilateral subcortical structures, mean
hemisphere thickness, and surface-area-weighted lobe thickness. The parser accepts
both long and short FreeSurfer measure keys so estimated intracranial volume is not
dropped.

The ontology is constructed from feature labels, descriptions, units, measurement
sources, and reference-only sample values. The LLM proposes semantic domains and
subdomains. Correlation clusters and target associations are descriptive quality
checks only and do not determine the hierarchy. Code enforces exact, non-redundant
leaf coverage.

`ontology/ontology_features.csv` is a separate benchmarking matrix. Its predictor
columns encode `DOMAIN|subdomain|feature`, while the outcome is explicitly prefixed
with `TARGET|`. That CSV is never an engine input. The engine input builder asserts
that the target and its three IST subscales are absent from predictor specifications.

## Cumulative and ablation tiers

| Tier | Included evidence | Features |
|---|---|---:|
| T1 | Demographics and physical | 6 |
| T2 | T1 + Personality | 11 |
| T3 | T2 + Motivation and affect | 16 |
| T4 | T3 + Identity and belief | 23 |
| T5 | T4 + Brain morphometry | 59 |
| T6 | T5 + Functional connectome | 87 |
| B1 | Brain morphometry only | 36 |
| B2 | Functional connectome only | 28 |
| B3 | Morphometry + connectome only | 64 |

T3 and T4 are cumulative. For example, T3 contains demographics, personality, and
motivation/affect together. T4 contains all of those plus identity/belief. This is why
their scores answer the requested incremental evidence question.

## Parallel execution, retries, and artifacts

Step 04 schedules all tier-participant pairs in one bounded process pool. Process
isolation prevents the engine settings singleton from mixing tier output paths.
Workers never write shared prediction files. The parent process atomically updates
one compact `predictions.json` per tier after each completed job, so interrupted runs
resume safely. Failed or schema-invalid predictions receive two retries with
exponential backoff by default.

Verbose engine reports are temporary unless `--keep-run-artifacts` is requested.
Canonical committed outputs are compact:

- `results/<tier>/predictions.json`
- `results/<tier>/metrics.json`
- `results/<tier>/rank_comparison.csv`
- `results/tiers_summary.json`

## Metrics and rank recovery

All headline tier metrics use the same common participant intersection: a participant
must have a valid numeric prediction in every evaluated tier. Each tier also records
attempted N, valid N, failures, success rate, and available-case metrics. This prevents
one tier from looking better only because a difficult participant failed elsewhere.

Point metrics are native IST MAE, median absolute error, RMSE, R2, Pearson correlation,
MAE in reference SD units, IQ-scale-equivalent MAE, and improvement over a prediction
that always uses the disjoint reference mean. MAE and Spearman receive paired
participant-bootstrap 95 percent intervals from 2,000 resamples.

Rank 1 is the highest score. Ties receive average ranks. Rank recovery includes:

- Spearman rho between actual and predicted scores;
- Kendall tau-b;
- pairwise ordering accuracy, with a predicted tie worth 0.5;
- mean absolute rank-position error and its percentile-range version;
- top-quartile and bottom-quartile overlap;
- one row per participant in `rank_comparison.csv` with actual rank, predicted rank,
  rank error, native scores, and IQ-equivalent scores.

The Spearman bootstrap interval measures uncertainty in rank recovery across
participants. It is not a measure of stochastic stability across repeated LLM runs.
Repeated-run stability would require multiple independent predictions per tier and
participant.

## Leakage boundary

The following controls are enforced:

- IST total, fluid, memory, and crystallized scores are excluded from predictors.
- Evaluation sampling does not use target values.
- Feature normalization is fitted only on non-evaluation participants.
- Target mean and SD shown to agents come only from the disjoint reference split.
- The observed evaluation target range is not shown to agents.
- Public source participant IDs are blinded in all model-visible artifacts.
- The semantic ontology uses reference-only feature samples.
- Headline tier comparisons use one common success cohort.

Aggregate target calibration is provided because the engine must know the numeric
scale on which to answer. It does not contain any evaluation participant's label.

## Reproduction

Use the project Python environment with the required scientific and LLM dependencies:

```bash
cd validation/datasets/AOMIC_ID1000/pipeline
/opt/anaconda3/bin/python 10_extract_freesurfer.py
/opt/anaconda3/bin/python 11_extract_connectome.py --ref-size 120 --workers 3
/opt/anaconda3/bin/python 01_explore_structure.py
/opt/anaconda3/bin/python 02_build_ontology.py
/opt/anaconda3/bin/python 03_build_compass_inputs.py
/opt/anaconda3/bin/python 04_run_compass.py --workers 12 --retries 2
/opt/anaconda3/bin/python 05_evaluate.py
/opt/anaconda3/bin/python ../notebooks/build_notebooks.py
```

The connectome step resumes completed subjects and deletes raw BOLD and mask files
after feature extraction. This bounds disk use while retaining the 28 derived
features and one 7 by 7 network matrix per participant.

## Limitations

This is a zero-shot LLM inference benchmark, not a trained predictive model or a
clinical validation. The AOMIC cohort is a narrow sample of healthy Dutch young
adults. IQ-equivalent values are descriptive transformations. LLM calls can vary and
provider failures can occur. Results must be interpreted with uncertainty intervals,
failure counts, baseline comparisons, and the full rank table.
