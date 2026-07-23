# Resting-State EEG Feature Extraction — First-Episode Psychosis

Reproducible participant-level feature extraction from eyes-open resting EEG in a
first-episode psychosis (FEP) vs. control cohort. The pipeline converts each raw
BrainVision recording into one row of **804 numeric EEG features** organized into
seven groups, plus separate quality-control and provenance artifacts.

**Scope:** research use only; sensor-space cortical topography, not source localization.
This is not a clinical diagnostic protocol.

## Data source

Two OpenNeuro datasets (same lab, same resting paradigm, different acquisition samples):

| Accession | Description | Subjects |
|---|---|---|
| [ds003944](https://openneuro.org/datasets/ds003944/versions/1.0.1) | FEP vs. Control Resting, sample 1 | 82 |
| [ds003947](https://openneuro.org/datasets/ds003947/versions/1.0.1) | FEP vs. Control Resting, sample 2 | 61 |

**The raw EEG (~19 GB) is intentionally not committed to git.** Download the two
snapshots into `data/ds003944/` and `data/ds003947/` (e.g. via the OpenNeuro CLI or
DataLad) to re-run extraction. The released feature matrix under `data/processed/`
is committed and is self-contained for analysis.

Source reference: Phalen et al. (2020), *Biol. Psychiatry: CNNI* 5(10):961–970.

## Deliverable

`data/processed/eeg_fep_rest_v1/`

| File | Contents |
|---|---|
| `features_eeg_804.csv` / `.parquet` | **143 participants × 804 features** (+ `participant_id`, `dataset`). QC-excluded subjects are present as blank rows. |
| `qc_subjects.tsv` | Per-subject quality metrics for all 143 (channels retained, epochs kept, ICA components removed, fit quality, etc.). |
| `exclusions.tsv` | The 11 QC-excluded subjects and reasons. |
| `feature_manifest.json` | Locked column order + SHA-256 hash. |
| `run_manifest.json` | Provenance: package versions, git commit, seeds, mitigations. |

### Feature groups (804 total)

| Group | Features | Description |
|---|---:|---|
| A — Spectral | 168 | Absolute/relative band power, band-power ratios, hemispheric asymmetry (delta–low gamma × 11 scopes) |
| B — Peak alpha frequency | 33 | Individual alpha peak frequency, center of gravity, prominence |
| C — Aperiodic (1/f) slope | 22 | specparam aperiodic exponent & offset |
| D — Entropy / complexity | 44 | Sample, permutation, spectral entropy; Lempel-Ziv complexity |
| E — Fractal | 22 | Higuchi fractal dimension; detrended fluctuation analysis |
| G — Functional connectivity | 225 | Debiased squared weighted phase-lag index (45 region pairs × 5 bands) |
| H — Graph theory | 290 | Global + nodal topology over density-thresholded connectivity graphs |

*(Group F, microstates, from the original specification is not implemented in this
version. Reviving it would be a separate `v2` schema.)*

Spatial scopes are one `global` summary plus 10 broad cortical scalp regions
(frontal / central / temporal / parietal / occipital × left / right). Features are
computed per channel, then reduced by region median; connectivity uses one frozen
per-subject spatial weighting per region built from non-interpolated channels only.

## Running the pipeline

```bash
cd code
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Python 3.12
# place raw data under ../data/ds003944 and ../data/ds003947, then:
.venv/Scripts/python -m pipeline_v1.run_extraction
```

The batch is parallelized across subjects (8 workers) and takes roughly 1.5–2 hours
for all 143 subjects. Per-subject failures are logged and skipped, never aborting the run.

## Method notes

- **Preprocessing:** positional channel reconciliation from `channels.tsv`, unit audit,
  harmonized resample to 250 Hz (native rates are mixed, 1000/3000 Hz), 0.5–45 Hz
  filtering, automated bad-channel detection (pyprep), ICA ocular/cardiac removal,
  average reference, 4 s epochs with automated rejection (autoreject).
- **QC gates** (subject excluded if failed): ≥120 s usable data, ≥30 clean epochs,
  ≥80% channels retained, ≤60% bad-epoch fraction, ≤20% interpolated. 11 of 143
  subjects were excluded here (8 for excessive bad channels, 3 for no usable epochs).
- **Missing values are preserved with a reason, never zero-filled** (e.g. a region with
  no detectable alpha peak). Populated rows are ~98% complete.
- **One flagged deviation for performance:** sample entropy (Group D) uses a fixed
  deterministic 40-epoch cap per channel rather than all retained epochs. Recorded in
  `run_manifest.json`.

## Layout

```
code/
  pipeline_v1/            # the extraction pipeline (importable package)
    run_extraction.py     # batch entry point
    schema.py             # locked 804-feature schema + manifest
    preprocessing.py, features/…   # stages and feature groups
  eeg_pipeline.py         # BrainVision loader helper (no-marker-file workaround)
  requirements.txt
data/
  ds003944/, ds003947/    # raw EEG — NOT in git; download from OpenNeuro
  processed/eeg_fep_rest_v1/   # committed feature matrix + QC + provenance
```
