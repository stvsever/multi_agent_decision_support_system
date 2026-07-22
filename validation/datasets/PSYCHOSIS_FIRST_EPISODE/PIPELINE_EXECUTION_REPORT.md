# Resting-State EEG Pipeline: Execution Report

First-episode-psychosis resting EEG, OpenNeuro ds003944 (n=82) and ds003947 (n=61). This report records
what was run and what it produced. The scientific methods are in `utils/PIPELINE_OVERVIEW.md`; the work is
reproduced by the three notebooks under `notebooks/` and the importable `utils/` package. Everything is
local; nothing was pushed to git.

## 1. What this run optimized over the previous pipeline

The previous pipeline worked but made several sub-optimal choices. This run rebuilt the analysis to
neuroscience-standard quality:

1. **One shared montage.** The two studies wired different 61-electrode subsets, and the old code used
   dataset-specific region maps, so a column like `occipital_left` meant different electrodes in each
   dataset (a montage-driven batch confound of diagnosis). Every feature is now computed on the 49
   electrodes shared by both datasets, with one region map.
2. **Bad channels interpolated, not dropped.** Flagged channels are interpolated by spherical spline
   before the recording is restricted to the shared 49, so every subject contributes the same electrodes.
3. **Canonical extractors.** The bespoke 1/f fit and label-blind microstates were replaced by standard
   tools: FOOOF/specparam (aperiodic exponent and offset, alpha peak), pycrostates (Koenig A-D
   microstates), and mne-connectivity (debiased wPLI). Feature column names were revised so each value
   matches its canonical definition.
4. **Interpretable values.** Every feature is delivered raw and as an age/sex control-referenced z-score.
5. **Clean structure.** All pipeline code moved into an importable `utils/` package; the top level holds
   only config, this report, and the data/results/notebooks/utils directories. Per-subject figure dumps
   were removed in favour of one representative-subject dashboard.

## 2. Cohort and preprocessing

- 143 recordings discovered and audited (all files complete, 64 channel rows each).
- All 143 harmonized onto the shared 49-channel montage, average-referenced.
- 134 recordings feature-eligible; 9 excluded by automated QC (retained epochs, bad-channel or bad-epoch
  fraction). Excluded recordings keep the schema with empty feature cells.
- Preprocessing reused the audited ICA-cleaned checkpoints and re-derived the harmonized montage, which is
  scientifically identical for the shared 49 channels; `python -m utils.run_preprocess --from-raw`
  recomputes the full ICA cleaning from the source binaries.

## 3. Feature extraction

- 836 features per recording in eight families (A spectral 168, B alpha peak 33, C aperiodic 22, D entropy
  44, E fractal 22, F microstates 32, G connectivity 225, H graph 290).
- Final matrix: **143 recordings x 837 columns** (recording id + 836 features).
- **110,229 finite values, 9,319 explicit missing, zero infinite.** The 9 ineligible recordings are the
  all-empty rows; the remaining missing values are frequency bands whose sparse phase coupling could not
  form a connected graph, so that band's Family H values are legitimately not estimable.
- Group microstate templates were fit once on GFP-peak topographies pooled across the eligible subjects
  and relabelled to canonical A-D.

## 4. Interpretable and phenotype tables

- `results/eeg_features.csv` - 836 raw features per recording.
- `results/eeg_features_zscores.csv` - age/sex control-referenced z-scores (dataset-specific control
  reference, centred age and sex, control residual standard deviation). Internal reference scores, not
  external norms.
- `results/non_eeg_features.csv` - 143 x 247: identifiers, 9 covariates, 235 clinical phenotype targets.
- `results/non_eeg_feature_dictionary.csv` - provenance and description of every non-EEG column.
- `results/group_distribution_statistics.csv` - per-feature Control vs Psychosis comparison with an
  automated normality gate (Shapiro-Wilk) selecting a Welch t-test or Mann-Whitney U rank test, and
  Benjamini-Hochberg FDR corrected within each feature family (correcting across all 836 at once would be
  far too strict).

## 5. Notebooks (all pre-run under notebooks/)

- **01_LOAD_PREPROCESS_AND_EXPLORE_EEG.ipynb** - loading, the montage harmonization, a live preprocessing
  demonstration, cohort QC, the channel-quality atlas, and a representative cleaned recording.
- **02_EXTRACT_AND_VISUALIZE_EEG_FEATURES.ipynb** - schema and completeness audit, the raw-vs-z-score
  interpretation, and one documented section per feature family (A through H) with its own rich figure,
  a cross-family group-difference map, the representative-psychosis dashboard, and the non-EEG overview.
- **03_ONTOLOGY_AND_COMPASS_LADDER.ipynb** - the ontology, the four COMPASS input files, the five-tier
  ladder, and the hierarchical diagnosis-plus-symptom results on 10 subjects.

## 6. COMPASS ladder (notebook 03)

Notebook 03 builds one master ontology over every predictor (836 EEG features placed by deterministic
path under a Resting EEG domain; the non-neural clinical predictors grouped into three domains by an
OpenRouter model), writes the four COMPASS input files per subject per tier with leakage-controlled
predictors and disjoint-reference z-scores, and runs the integrated hierarchy (diagnosis, then BPRS total,
then SAPS and SANS globals) on 10 subjects (5 psychosis, 5 control) across the five tiers.

The ontology is emitted as the full linguistic representation under `results/compass/ontology/`: the
machine-readable `subclass_structure.json`, a Protege-loadable `psychosis_fep.owl`, a self-contained
interactive `ontology_viewer.html`, a hierarchy-encoded `ontology_features.csv`, and the
`exploration_report.json` and `ontology_report.json` quality reports (935 leaves, four domains, max depth
five; the QA records the semantic-vs-statistical cluster agreement and an LLM MECE review). The ladder is
re-runnable from the notebook with a single toggle, or from the command line with `python -m
utils.run_ladder`.

- 50 runs (10 subjects x 5 tiers), all succeeded. OpenRouter spend for the run: about 1.90 USD with the
  small demo model.
- 90 columns are excluded from every tier as leakage controls; the lean-EEG set is 29 psychosis-implicated
  features.
- Diagnosis balanced accuracy along the ladder: T1 demographics 0.50 (chance), T2 clinical profile 0.60
  (AUROC 0.68), T3 full multimodal 0.50, T4 EEG lean 0.30, T5 EEG rich 0.40.

On 10 subjects with a small model this is illustrative, not a validation: the non-neural clinical profile
(T1 to T2) carries most of the recoverable diagnosis signal, while EEG-only tiers are near chance at this
sample size. The value of the notebook is the working ingestion-to-prediction pipeline and the rich
hierarchical visualizations of every node. The full-cohort run is one command
(`python -m utils.run_ladder`).

## 7. Interpretation boundaries

- Features are cortical sensor-space summaries. They are not source-localized, not subcortical, and make
  no clinical diagnosis.
- Automated QC only; no manual annotation.
- The group figures contrast Control and Psychosis descriptively; they are not a classifier.
- The 10-subject ladder is a demonstration of the ingestion and prediction pipeline with a small model;
  its metrics are illustrative, not a validation.
- Microstate class letters follow the canonical Koenig A-D scheme by topographic orientation; they are a
  reproducible relabelling of data-derived group templates.

## 8. Reproducing

```bash
python -m utils.run_preprocess      # harmonized preprocessing (add --from-raw for full ICA)
python -m utils.run_features        # group microstate templates + 836 features/subject
python -m utils.interpretable       # non-EEG table, dictionary, control z-scores
python -m utils.run_ladder          # ontology + four input files + 10-subject tier ladder
```

No em dashes are used in the authored deliverables. No data was pushed to git.
