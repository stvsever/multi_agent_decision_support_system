# AOMIC-ID1000 intelligence inference validation

This folder validates COMPASS on the ID1000 component of the Amsterdam Open MRI
Collection. The source is [OpenNeuro ds003097](https://openneuro.org/datasets/ds003097),
version 1.2.1, DOI `10.18112/openneuro.ds003097.v1.2.1`, released under CC0.
The accompanying data descriptor is [Snoek et al., 2021](https://doi.org/10.1038/s41597-021-00870-6).

The prediction task is zero-shot, participant-level regression of the native
`IST_intelligence_total` composite. The detailed protocol, leakage controls,
target transformation, tier definitions, metrics, and reproduction commands are in
[METHODOLOGY.md](METHODOLOGY.md).

## What ID1000 contains

ID1000 is the population-oriented AOMIC cohort. Researchers recorded 992 people and
released 928 after quality control. Participants were healthy Dutch young adults,
ages 19 to 26, recruited to approximate the Dutch population's educational-level
distribution within that age range. Collection took place from 2010 to 2012 on a
Philips 3 T Intera scanner with a 32-channel head coil.

The public dataset contains:

| Data family | Contents |
|---|---|
| Structural MRI | Three T1-weighted anatomical acquisitions per participant |
| Diffusion MRI | Three diffusion-weighted acquisitions per participant, where available |
| Functional MRI | One approximately 10 minute 38 second BOLD run while viewing a non-narrative compilation of 22 scenes from *Koyaanisqatsi* |
| Physiology | Cardiac pulse and respiratory-belt traces recorded during fMRI |
| Demographics | Age, self-reported biological sex, handedness, BMI, education level, and parental socioeconomic background |
| Identity and belief | Felt male and female identity, attraction to males and females, religious upbringing, current religion, and religious importance |
| Intelligence | IST 2000-R total, memory, fluid, and crystallized scores |
| Personality | NEO-FFI openness, conscientiousness, extraversion, agreeableness, and neuroticism |
| Motivation and affect | BIS, BAS drive, BAS fun seeking, BAS reward responsiveness, and trait anxiety |
| Acquisition metadata | Per-participant diffusion repetition times and BIDS metadata |
| Task materials | Movie stimulus and annotated timing information |
| Derivatives | BIDS-organized preprocessed structural, diffusion, and functional products, including FreeSurfer and fMRIPrep derivatives with quality-control information |

The wider AOMIC collection also includes PIOP1 and PIOP2. Their resting-state and
task-fMRI paradigms are not part of ID1000. This validation uses only ID1000.

## Data used by this validation

The source dataset is much richer than the final predictor matrix. This experiment
uses 87 interpretable, non-cognitive predictors:

| Block | Features | Construction |
|---|---:|---|
| Demographics and physical | 6 | Directly from `participants.tsv` |
| Personality | 5 | NEO-FFI scale totals |
| Motivation and affect | 5 | BIS/BAS and trait anxiety |
| Identity and belief | 7 | Gender identity, attraction, and religion variables |
| Brain morphometry | 36 | Global volumes, bilateral subcortical volumes, and surface-area-weighted cortical thickness summaries from FreeSurfer |
| Functional connectome | 28 | Seven within-network and 21 between-network correlations from movie-fMRI using Schaefer-100 and Yeo-7 |

The validation does not currently use diffusion MRI, physiological traces, raw task
timing, voxelwise morphometry, cortical surface area as a separate regional feature
family, or parcel-level connectome edges. These remain possible extensions.

The target and all three IST subscales are excluded from predictors. MRI acquisition
parameters are also excluded because they are not participant phenotypes.

## Tiers

| Tier | Included evidence | Features |
|---|---|---:|
| T1 | Demographics and physical | 6 |
| T2 | T1 plus personality | 11 |
| T3 | T2 plus motivation and affect | 16 |
| T4 | T3 plus identity and belief | 23 |
| T5 | T4 plus brain morphometry | 59 |
| T6 | T5 plus functional connectome | 87 |
| B1 | Brain morphometry only | 36 |
| B2 | Functional connectome only | 28 |
| B3 | Morphometry plus connectome only | 64 |

T4 is the requested combined self-report tier. It contains personality, motivation
and affect, and identity and belief together, along with the demographic baseline.

## Current run status

The locked evaluation cohort contains 100 randomly selected eligible participants.
Selection is target-blind, model-visible participant IDs are blinded, and all
normalization and target calibration use participants outside the evaluation cohort.

The current saved run used `google/gemini-3.1-flash-lite` with one inference iteration:

| Tier | Valid predictions |
|---|---:|
| T1, T2, T3, T4, T5, T6 | 100 of 100 each |
| B1, B2 | 100 of 100 each |
| B3 | 74 of 100 |

This is 874 valid predictions of the planned 900. B3 contains 12 recorded failed
attempts and 14 participants that were not attempted before provider credits were
exhausted. Headline metrics use the same 74-person common-success intersection across
all nine tiers. The result files identify attempted N, valid N, failed N, confidence
intervals, baseline performance, and rank-recovery statistics. Resume step 04 with
the same model after replenishing provider credit, then rerun step 05 to obtain the
final 100-person common comparison.

## Folder contents

| Path | Purpose |
|---|---|
| `dataset/` | BIDS participant table and field descriptions |
| `brain/` | Derived morphometry and network-connectivity features |
| `ontology/` | Exploration report, semantic ontology, OWL file, flat benchmark matrix, and interactive viewer |
| `compass_inputs/` | Blinded, tier-specific inputs for the locked cohort |
| `pipeline/` | Extraction, ontology, input-building, inference, and evaluation scripts |
| `results/` | Locked cohort metadata, compact predictions, metrics, and rank comparisons |
| `notebooks/` | Reproducible exploratory and result visualizations |

Dataset-specific configuration stays in `pipeline/config.py`. Reusable extraction,
ontology, input-writing, and evaluation logic stays under `validation/common/`.

## Reproduction

```bash
cd validation/datasets/AOMIC_ID1000/pipeline
/opt/anaconda3/bin/python run_tiers.py --skip-brain
/opt/anaconda3/bin/python run_tiers.py --live --skip-brain --workers 12
```

See [METHODOLOGY.md](METHODOLOGY.md) before interpreting or extending the benchmark.
