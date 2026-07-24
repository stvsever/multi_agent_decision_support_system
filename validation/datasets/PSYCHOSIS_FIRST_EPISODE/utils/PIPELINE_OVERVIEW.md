# Resting-State EEG-Based Feature Extraction Pipeline

**Datasets:** OpenNeuro [ds003944 v1.0.1](https://openneuro.org/datasets/ds003944/versions/1.0.1) and
[ds003947 v1.0.1](https://openneuro.org/datasets/ds003947/versions/1.0.1), University of Pittsburgh
first-episode-psychosis resting EEG (eyes open, ~5 min, 1000 Hz, EEG recorded on an Elekta Neuromag
system with an EasyCap layout).

This document describes the science of the pipeline: how the two recordings are loaded, harmonized onto
a single montage, cleaned, and reduced to 836 canonical resting-EEG features whose column names match
their definitions exactly. It is the methods companion to `PIPELINE_EXECUTION_REPORT.md` (the run log)
and the two notebooks under `notebooks/`.

The pipeline stops at validated feature values. It contains no classifier and makes no clinical claim.
The features are cortical sensor-space summaries, not source-localized or subcortical estimates.

---

## 1. Cohort

| Dataset  | Recordings | Control | Psychosis | Online reference | Distinct electrodes |
|----------|-----------:|--------:|----------:|------------------|--------------------:|
| ds003944 | 82         | 32      | 50        | TP9              | 61 (incl. M2)       |
| ds003947 | 61         | 30      | 31        | TP9              | 61 (incl. M2)       |
| **Total**| **143**    | **62**  | **81**    |                  |                     |

Both cohorts mix cases and controls, so dataset is only a partial confound of diagnosis. The two studies
wired slightly different 61-electrode subsets; this is the single most important thing the pipeline has
to reconcile.

---

## 2. Loading (utils/io.py)

The BrainVision files were exported from EEGLAB and store only generic labels (`EEG001 ... EEG064`) in
the `.vhdr` header. The true 10-10 names and channel types live in the row-aligned `*_channels.tsv`
sidecar. MNE's BrainVision reader would keep the generic labels, so the loader reads the VECTORIZED
float32 binary directly and attaches the sidecar names positionally. This is the only load path that
yields correctly named electrodes for this dataset. Sampling frequency, data orientation, binary format,
and sample count are all validated against the header and JSON sidecar before a recording is accepted.

---

## 3. Harmonization onto a shared montage (utils/montage.py)

The two montages intersect in exactly **49 cortical electrodes** (the M2 mastoid and the online
reference are excluded). Every regional, global, connectivity and graph feature is computed on this one
shared 49-channel montage, so a column such as `...__alpha_8_13_hz__occipital_left` addresses the same
electrodes in both datasets rather than being confounded by layout. This replaces the previous approach,
which used dataset-specific region maps and therefore mixed different electrodes into the same column.

Ten balanced lateralized regions of interest are defined once over the shared montage; the four midline
electrodes (Fz, Cz, Pz, Oz) contribute to the global scope only, so left/right asymmetry contrasts stay
strictly lateralized.

| Region | Electrodes |
|--------|-----------|
| frontal_left | Fp1, AF7, AF3, F7, F5, F3, F1 |
| frontal_right | Fp2, AF4, F2, F4, F6, F8 |
| central_left | FC5, FC1, C5, C3, C1 |
| central_right | FC6, FC2, C6, C4, C2 |
| temporal_left | FT7, T7, TP7 |
| temporal_right | FT8, T8, TP8 |
| parietal_left | CP3, CP1, P7, P5, P3 |
| parietal_right | CP4, CP2, P4, P6, P8 |
| occipital_left | PO7, PO3, O1 |
| occipital_right | PO8, PO4, O2 |

The eleven scopes are `global` plus these ten regions. The ten regions are also the nodes of the
connectivity and graph analyses.

---

## 4. Preprocessing (utils/preprocess.py)

Per recording:

1. crop to a common analysable window (300 s, 3 s trimmed each end);
2. band-pass 0.5-45 Hz (FIR, zero-phase) and resample to 250 Hz;
3. flag bad cortical channels (flat, robust-variance outlier, low neighbour correlation, residual line
   noise);
4. remove ocular and cardiac components with picard extended-infomax ICA guided by the VEOG and ECG
   channels (at most 8 components);
5. **harmonize**: interpolate the flagged channels by spherical spline on the native montage, restrict
   to the 49 shared electrodes, and re-reference to their average;
6. cut fixed 4 s epochs and score each for absolute amplitude, robust amplitude, and 30-45 Hz muscle
   power.

Interpolating before restricting to the shared 49 means every subject contributes the same electrodes to
every feature. A recording is **feature-eligible** when it retains at least 30 clean epochs (>=120 s),
at most 20 percent bad channels, and at most 60 percent bad epochs.

The 60 Hz mains line is removed by the 45 Hz low-pass rather than a notch. The average reference is
computed on the final 49-channel set.

---

## 5. Grouped feature extraction (utils/features.py)

836 features in eight families. Groups B, C, F and G are computed with the standard neuroscience
packages (`fooof`/specparam, `pycrostates`, `mne-connectivity`) so each value matches the canonical
definition its name promises. Every measure is summarized across the eleven scopes, or across the ten
region nodes, exactly as its column name states.

### 5.1 Family A - Spectral (168)
Multitaper power spectral density (`psd_array_multitaper`, 1.5 Hz bandwidth) over the five bands
delta (1-4), theta (4-8), alpha (8-13), beta (13-30), low gamma (30-45).
- `log10_absolute_power_uv2` and `relative_power_fraction_of_1_45_hz`, per band per scope (5 x 11 x 2 = 110);
- `natural_log_power_ratio` for theta/alpha, theta/beta, alpha/delta, per scope (3 x 11 = 33);
- `log10_power_asymmetry_right_minus_left` per band per region (5 x 5 = 25).

### 5.2 Family B - Alpha peak (33)
FOOOF (`fooof`) parameterizes each channel spectrum over 1-45 Hz; the strongest periodic peak with a
centre in 7-14 Hz gives the alpha peak. Per scope: `center_frequency_hz`, `power_log10_uv2_above_aperiodic`,
`bandwidth_hz` (3 x 11 = 33). These are canonical FOOOF Gaussian parameters, replacing the previous
bespoke peak detector.

### 5.3 Family C - Aperiodic 1/f (22)
From the same FOOOF fit, per scope: `exponent` and `offset_log10_uv2` (2 x 11 = 22). This replaces the
previous hand-rolled log-log line fit with the standard spectral-parameterization aperiodic model.

### 5.4 Family D - Entropy and complexity (44)
Per channel per epoch (median across epochs), via `antropy`: `sample_entropy`,
`permutation_entropy_normalized`, `spectral_entropy_normalized`, `lempel_ziv_complexity_normalized`,
each per scope (4 x 11 = 44).

### 5.5 Family E - Fractal (22)
`higuchi_fractal_dimension` (per epoch, median) and `detrended_fluctuation_exponent` (DFA on the
continuous cleaned trace, capturing long-range temporal correlations), each per scope (2 x 11 = 22).

### 5.6 Family F - Microstates (32)
`pycrostates` modified K-means fits four group templates on GFP-peak topographies pooled across all
eligible subjects; the four maps are relabelled to the canonical Koenig A-D scheme by their topographic
orientation (`utils.montage.canonical_microstate_order`), so `class_a ... class_d` carry their
conventional meaning. Each subject is back-fit to these shared templates.
- per class: `mean_duration_ms`, `coverage_fraction`, `occurrence_per_second`,
  `global_explained_variance_fraction` (4 x 4 = 16);
- `transition_probability` for the 12 ordered class pairs (self-transitions removed, then row-normalized);
- global: `global_explained_variance_fraction`, `transition_entropy_normalized`,
  `sequence_lempel_ziv_complexity_normalized`, `mean_global_field_power_uv` (4).

### 5.7 Family G - Functional connectivity (225)
`mne-connectivity` debiased weighted phase-lag index squared (`wpli2_debiased`, multitaper) between the
ten region nodes (each node is the first within-region principal component per epoch), per band, for the
45 region pairs (5 x 45 = 225). wPLI is insensitive to zero-lag volume conduction.

### 5.8 Family H - Graph theory (290)
Each band's 10-node wPLI matrix is thresholded across densities 20-50 percent (maximum spanning tree plus
strongest edges), and metrics are integrated as an area under the density curve.
- global (8 per band x 5 = 40): `mean_edge_weight`, and AUC of global efficiency, characteristic path
  length, mean clustering, transitivity, modularity Q, assortativity, small-world propensity;
- node (5 per band per node x 5 x 10 = 250): `strength_normalized`, and AUC of local efficiency,
  betweenness, eigenvector centrality, participation coefficient.

Small-world propensity uses 100 degree-preserving random surrogates and a matched lattice (Muldoon 2016).

**Family totals:** A 168, B 33, C 22, D 44, E 22, F 32, G 225, H 290 = **836**.

When a band's connectivity graph cannot reach the requested density (sparse phase coupling, usually in
low gamma), that band's Family H values are left missing rather than forced, and the subject-level QC
records which bands were unavailable.

---

## 6. Interpretable deliverables (utils/interpretable.py)

- **`results/eeg_features.csv`** - the 836 raw feature values, one row per recording.
- **`results/eeg_features_zscores.csv`** - each feature as an age/sex control-referenced z-score. Within
  each dataset, the control group is regressed on centred age and sex; every subject's residual is
  divided by the control residual standard deviation. This answers "how far is this person from a
  same-age, same-sex control?" and keeps the two acquisitions on their own reference. These are internal
  reference scores, not external population norms.
- **`results/non_eeg_features.csv`** - identifiers, 9 covariates (demographics, Hollingshead SES,
  chlorpromazine equivalent), and 235 clinical phenotype targets (BPRS, SANS, SAPS, GAF/GAS, SFS,
  MATRICS, WASI), harmonized across both datasets and positioned to the right.
- **`results/non_eeg_feature_dictionary.csv`** - provenance, description, units, coverage per column.

---

## 7. What was upgraded relative to the previous pipeline

- One shared 49-channel montage for both datasets (was dataset-specific region maps -> layout confound).
- Bad channels interpolated before restriction (was dropped, breaking a fixed feature schema).
- Canonical aperiodic and alpha-peak parameters from FOOOF (was a bespoke log-log line fit).
- Canonical Koenig A-D microstate labels from pycrostates (was label-blind, non-canonical letters).
- Canonical debiased wPLI from mne-connectivity (was a manual implementation).
- Feature column names renamed where needed so every value matches its definition.

---

## 8. Reproducing

```bash
python -m utils.run_preprocess     # harmonized preprocessing for all recordings
python -m utils.run_features       # group microstate templates + 836 features/subject
python -m utils.interpretable      # non-EEG table, dictionary, control z-scores
```

`--from-raw` on the preprocessing step runs the full ICA cleaning from the source binaries; the default
reuses the audited ICA-cleaned checkpoints and only re-derives the harmonized montage, which is
scientifically identical for the shared 49 channels and much faster. Notebook 01 covers preprocessing
and exploration; notebook 02 covers extraction and the per-family visualizations.
