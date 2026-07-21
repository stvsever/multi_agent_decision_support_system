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
uses 279 interpretable, non-cognitive predictors, organised under one arbitrary-depth
ontology (`Brain` splits into `Morphometry` and `Connectomics`, each with deeper
region/network structure):

| Block | Features | Construction |
|---|---:|---|
| Demographics and anthropometrics | 6 | Directly from `participants.tsv` |
| Psychological constructs | 10 | NEO-FFI Big Five, BIS/BAS reinforcement sensitivity, STAI trait anxiety |
| Identity and belief systems | 7 | Gender identity, attraction, and religion variables |
| Brain > Morphometry | 228 | Per-region Desikan-Killiany cortical thickness, surface area and gray-matter volume (34 regions x 2 hemispheres each, grouped by lobe), plus 16 subcortical volumes and 6 global volumetric summaries, from FreeSurfer |
| Brain > Connectomics | 28 | Seven within-network and 21 between-network correlations from movie-fMRI using Schaefer-100 and Yeo-7 |

The morphometry branch is high-resolution: rather than collapsing cortex to six lobe
summaries (the previous 36-feature version), it exposes the full per-region atlas the
FreeSurfer pre-processing already produces (228 leaves). Re-extracting the connectome
at a finer atlas (e.g. Schaefer-200 / 17 Yeo sub-networks, 153 edges) is a one-line
config change in `validation/common/connectome.py`; the cached derivatives here are
7-network.

The validation does not currently use diffusion MRI, physiological traces, raw task
timing, voxelwise morphometry, or parcel-level connectome edges. These remain possible
extensions and slot into the same ontology without engine changes.

The target and all three IST subscales are excluded from predictors. MRI acquisition
parameters are also excluded because they are not participant phenotypes.

## Tiers

| Tier | Included evidence | Features |
|---|---|---:|
| T1 | Demographics and anthropometrics | 6 |
| T2 | T1 plus personality | 11 |
| T3 | T2 plus motivation and affect | 16 |
| T4 | T3 plus identity and belief | 23 |
| T5 | T4 plus brain morphometry | 251 |
| T6 | T5 plus functional connectome | 279 |
| B1 | Brain morphometry only | 228 |
| B2 | Functional connectome only | 28 |
| B3 | Morphometry plus connectome only | 256 |

T4 is the complete self-report tier. It combines demographics, personality,
motivation and affect, and identity and belief in one cumulative feature set.
T5 and T6 add the high-resolution brain modalities.

## Results

The locked evaluation cohort contains 100 randomly selected eligible participants.
Selection is target-blind, model-visible participant IDs are blinded, and all
normalization and target calibration use participants outside the evaluation cohort.

### A. Prior full run: 100 subjects across all nine tiers (earlier feature structure)

Saved run with `google/gemini-3.1-flash-lite`, one inference iteration, metrics on the
74-person common-success intersection across all tiers (native IST rank recovery; MAE
on the 100/15 IQ-equivalent scale). Feature counts are the earlier lobe-level structure.

| Tier | n_feat | Pearson r | Spearman rho | MAE (IQ-equiv) |
|---|---:|---:|---:|---:|
| T1 demographics | 6 | 0.48 | 0.49 | 11.7 |
| T2 + personality | 11 | 0.43 | 0.45 | 12.2 |
| T3 + motivation/affect | 16 | 0.48 | 0.46 | 11.8 |
| T4 + identity/belief | 23 | 0.34 | 0.25 | 12.6 |
| T5 + morphometry | 59 | 0.47 | 0.45 | 11.5 |
| T6 full multimodal | 87 | 0.41 | 0.42 | 11.8 |
| B1 morphometry only | 36 | 0.39 | 0.45 | 12.0 |
| B2 connectome only | 28 | 0.01 | -0.01 | 13.7 |
| B3 brain only | 64 | 0.29 | 0.32 | 12.8 |

**Interpretation (and the psychometric-leakage question).** Demographics alone (T1:
age, sex, handedness, BMI, education, socio-economic background) already recover rank
at Spearman ~0.49. Education and SES are well-known population correlates of measured
intelligence, so this is a real proxy signal rather than target leakage; adding the
self-report psychometric tiers (T2-T4) does not improve on it, so there is no evidence
that personality/BIS-BAS/anxiety items are inflating the result. The scientifically
interesting finding is that **brain-only** evidence, with no self-report at all, still
recovers a moderate ranking: morphometry alone reaches Spearman ~0.45 (B1) and the full
brain-only tier ~0.32 (B3). The coarse 7-network **connectome alone carries essentially
no individual-differences signal** (B2 ~0), consistent with the literature that IQ
prediction from functional connectivity needs finer, edge-level parcellation - which is
exactly why the connectome atlas resolution is now a config knob.

### B. Fresh full-engine run on the upgraded high-resolution structure (2 subjects)

To keep cost low while confirming the upgraded 279-feature pipeline runs end to end, two
blinded subjects were pushed through the full multimodal tier (T6). Results in
`results/full_engine_2subject/`:

| Subject | Ground truth (IST) | Predicted | ~True IQ | ~Pred IQ |
|---|---:|---:|---:|---:|
| eval-0001 | 227 | 192.5 | 110 | 97 |
| eval-0002 | 267 | 232.5 | 125 | 112 |

The two subjects were **ranked correctly** (eval-0002 > eval-0001) and the predicted gap
(40.1 IST) almost exactly matched the true gap (40.0 IST). Both were underestimated by
about 13 IQ-equivalent points, the conservative regression-to-the-mean bias expected when
inferring two above-average subjects blind from non-cognitive evidence only. This is a
two-subject demonstration, not a statistical result; the prior 100-subject run above is
the quantitative benchmark.

## Folder contents

| Path | Purpose |
|---|---|
| `dataset/` | BIDS participant table and field descriptions |
| `brain/` | Derived morphometry and network-connectivity features |
| `ontology/` | Exploration report, arbitrary-depth ontology, OWL file, flat benchmark matrix, and interactive viewer |
| `compass_inputs/` | Blinded, tier-specific inputs for the locked cohort |
| `pipeline/` | Extraction, ontology, input-building, inference, and evaluation scripts |
| `results/` | Locked-cohort metadata, per-tier predictions/metrics, and `full_engine_2subject/` (fresh high-resolution run) |
| `notebooks/` | Reproducible exploratory and result visualizations (end to end: tabular, brain, ontology + results) |

Dataset-specific configuration stays in `pipeline/config.py`. Reusable extraction,
ontology, input-writing, and evaluation logic stays under `validation/common/`. The
generic, modality-agnostic ingestion engine (`validation/common/ingest.py`) turns any
pre-processed feature table into `loaded/subject_NNN/` COMPASS inputs and is what makes
adding a new modality (lesion masks, EEG, ...) a matter of writing a small feature
adapter, not touching the engine.

## Reproduction

```bash
cd validation/datasets/INTELLIGENCE/pipeline
/opt/anaconda3/bin/python run_tiers.py --skip-brain
/opt/anaconda3/bin/python run_tiers.py --live --skip-brain --workers 12
```

See [METHODOLOGY.md](METHODOLOGY.md) before interpreting or extending the benchmark.
