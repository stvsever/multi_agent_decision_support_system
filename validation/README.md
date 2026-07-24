# COMPASS validation framework

This directory contains reproducible evaluations of COMPASS on open-access,
multimodal datasets. It provides a consistent workflow for data exploration,
semantic ontology construction, participant encoding, inference, and quantitative
evaluation while keeping the core engine dataset-agnostic.

Dataset-specific configuration, derived features, model inputs, and results remain
inside each dataset directory. Reusable validation components live in `common/`.

## Structure

```text
validation/
  README.md                    framework overview and dataset registry
  common/                      reusable, dataset-agnostic validation components
    explore.py                   automated data understanding (types, correlations,
                                 feature clustering, redundancy, quality flags)
    manifest.py                  deterministic feature-structure profiling
    ontology.py                  arbitrary-depth ontology (path hints + semantic LLM
                                 grouping) + QA review + OWL/JSON/CSV
    viewer.py                    self-contained interactive ontology explorer (HTML)
    deviation.py                 reference strategies (cohort / external / absolute)
    compass_writer.py            emits the four COMPASS participant files (full depth)
    ingest.py                    generic modality-agnostic ingestion CLI: features ->
                                 loaded/subject_NNN/ (subject detection, free-text, the LLM
                                 reads all column names + data-dictionary to build the ontology)
    freesurfer.py                high-resolution FreeSurfer morphometry extraction
    connectome.py                movie-fMRI network connectome extraction
    tiers.py                     project the ontology onto a complexity tier
    llm.py                       minimal OpenRouter client (reads repo-root .env)
  datasets/
    INTELLIGENCE/             OpenNeuro ds003097 (AOMIC-ID1000) validation
      README.md                  dataset overview, features, tiers, and results
      METHODOLOGY.md             cohort, blinding, leakage, and metrics protocol
      dataset/                   CC0 participant table and data dictionary
      pipeline/                  extraction, ontology, inference, and evaluation
      brain/                     derived FreeSurfer and connectome features
      ontology/                  OWL, hierarchy, benchmark matrix, and QA reports
      compass_inputs/<tier>/     blinded inputs per tier and participant
      results/<tier>/            compact predictions, metrics, and rank tables
      notebooks/                 reproducible exploration and result visualizations
```

## How ingestion works

1. **Explore**: profile types, distributions, missingness, correlations, feature
   clusters, near-duplicates, target associations, and data-quality flags.
2. **Ontologize**: organize all features into a strict, non-redundant, **arbitrary-depth**
   subclass hierarchy (`Phenotype Feature -> DOMAIN -> ... -> FEATURE`) by **meaning**,
   never by statistics (two unrelated measures can correlate by accident). Two placement
   strategies mix freely: high-resolution modalities carry explicit `path` hints for a
   clean, deep, reproducible structure (e.g. `Brain -> Morphometry -> Cortical Thickness
   -> Frontal lobe -> region`), while un-pathed features are grouped semantically by the
   LLM (each feature's label, description, units, source and sample values given as TOON;
   the model proposes the domains, then organises each into subdomains in parallel). Code
   enforces exact, non-redundant coverage at any depth. An optional free-text
   user-guidance argument is injected into every prompt (the UI hook). The result is
   quality-assessed (adjusted Rand index against the statistical clusters, plus a compact
   LLM MECE review). Built **once per dataset** and reused as a fixed base template for
   every subject. Emits Protege-loadable OWL, a subclass JSON, a hierarchy-encoded
   benchmark CSV, an interactive HTML explorer, and QA reports.
3. **Encode**: standardize each feature into a deviation score with its label
   preserved. Reference strategy is auto-selected: `cohort` (batch is its own
   reference), `external` (supplied norms), or `absolute` (no reference / single
   subject, with an optional LLM range estimate). Deviation z-scores are meaningful only
   with a reference; in absolute mode the raw value carries the signal instead.
4. **Write**: render the four participant files, each mirroring the ontology at full
   depth. `hierarchical_deviation_map.json` carries the aggregated deviation signal at
   **every** level of the tree; `multimodal_data.json` holds the actual values at the
   leaves (so the two files do not duplicate content); `data_overview.json` reports
   coverage and token budget **per hierarchical group**, not just per domain.

The generic `ingest.py` CLI runs steps 2-4 for any modality-agnostic feature table:
it auto-detects the N subjects, writes one clean `loaded/subject_001/ ... /subject_00N/`
folder each, auto-detects a per-subject free-text note and folds it into the text
modality, and honours the `--reference-mode`, `--ontology`/`--build-ontology`, and
`--limit` mode switches so ingestion is explicit and bounded.

The engine reasons over the token-efficient TOON representation of the data. Feature
labels, units, hierarchy, reference deviations, and coverage metadata travel with
each value so the multi-agent system can interpret and audit the evidence.

## Why a clean hierarchical ontology matters to the engine

The engine's first move is a data-overview / exploration step, and its tools operate
per domain and per subtree (unimodal compression, multimodal narrative, coverage
tracking, chunked evidence). A flat feature list would force every tool to re-derive
structure; a clean, non-redundant subclass ontology gives the orchestrator explicit
domains to prioritise, lets the executor process independent subtrees in parallel,
and lets the predictor attribute evidence to a named branch. The ontology is the
scaffold that makes multi-modal evidence tractable and auditable.

## Datasets

Three OpenNeuro datasets, each with its own clinical phenotype and data-complexity ladder. The exact
phenotype output structure and tier ladder for each are in its `PHENOTYPE_AND_TIERS.md`.

| Dataset | Source | Clinical phenotype (prediction structure) | Modalities | Status |
|---|---|---|---|---|
| [AOMIC-ID1000](datasets/INTELLIGENCE/) | OpenNeuro ds003097 (CC0) | Total intelligence (univariate) then 3 IST subscales (multivariate) | Self-report, high-resolution FreeSurfer morphometry, fMRI connectome (279-feature structure) | 100-subject 9-tier run; fresh 2-subject full-ladder run |
| [First-Episode Psychosis](datasets/PSYCHOSIS_FIRST_EPISODE/) | OpenNeuro ds003944 + ds003947 | Diagnosis (binary) then BPRS total (univariate) then SAPS/SANS globals (multivariate) | Resting EEG (836 features, 8 families) plus demographics, socio-economic status, cognition/IQ, observed functioning | 143 recordings on one harmonized 49-channel montage; 5-tier ladder |
| [Numeracy after Stroke](datasets/NUMERACY_STROKE/) | OpenNeuro ds006533 (CC0) | Approximate and precise numeracy (two dissociable univariate phenotypes) | Demographics, aphasia severity, whole-brain and per-region lesion overlap (cortex/subcortex/cerebellum) | 105 left-hemisphere stroke survivors; 4-level lesion-resolution ladder |

**Short summary of the phenotype prediction structures and ladders**

- **INTELLIGENCE**: a two-level hierarchy - the root regresses total IST intelligence, and a child node
  jointly regresses the three IST subscales (fluid, memory, crystallised). Ladder T1 demographics -> T6
  brain connectome (six cumulative tiers plus brain-only probes).
- **PSYCHOSIS**: a mixed three-level hierarchy - a binary diagnosis at the root, BPRS total severity beneath
  it, then the SAPS positive-symptom and SANS negative-symptom globals beneath that. Ladder T1 demographics/SES
  -> T5 full 836-feature resting EEG.
- **NUMERACY**: two dissociable numeracy phenotypes (approximate non-symbolic vs precise symbolic), each a
  univariate regression on a population Z-score, predicted separately so their differential relationship to
  language/lesion features is the readout. Ladder T1 demographics -> T3 per-parcel lesion overlap.

The master notebook `validation_with_openneuro_datasets.ipynb` runs all three end to end: it load-checks
every record, runs a 2-subjects-per-dataset subset on a substantial tier, visualizes recovered phenotypes
against ground truth, and estimates the full-cohort API cost and wall time per dataset. It is resumable:
each run caches the moment it finishes and a re-run continues from where it stopped, so an interrupted or
quit batch never re-spends on completed runs.

More datasets plug in by copying a dataset folder and editing its config. The three show two supported
layouts: `INTELLIGENCE/` and `NUMERACY_STROKE/` use a `pipeline/` of numbered scripts, while
`PSYCHOSIS_FIRST_EPISODE/` uses an importable `utils/` package with three pre-run notebooks (loading and
preprocessing, feature extraction and visualization, and ontology plus the COMPASS tier ladder). All
reuse `common/` and the same four-file COMPASS contract. Large raw and processed EEG/MRI data stay local
(git-ignored); the derived feature tables, figures and COMPASS inputs are tracked.

## Requirements

`OPENROUTER_API_KEY` in the repository-root `.env`. LLM steps use a small model
(`google/gemini-3.1-flash-lite`) by default. Brain extraction needs `nilearn`,
`nibabel`, and network access to the OpenNeuro S3 mirror.
