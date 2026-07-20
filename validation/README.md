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
    ontology.py                  semantic LLM ontology + QA review + OWL/JSON/CSV
    viewer.py                    self-contained interactive ontology explorer (HTML)
    deviation.py                 reference strategies (cohort / external / absolute)
    compass_writer.py            emits the four COMPASS participant files
    freesurfer.py                FreeSurfer morphometry feature extraction
    connectome.py                movie-fMRI network connectome extraction
    tiers.py                     project the ontology onto a complexity tier
    llm.py                       minimal OpenRouter client (reads repo-root .env)
  datasets/
    AOMIC_ID1000/              OpenNeuro ds003097 validation
      README.md                  dataset overview, features, tiers, and run status
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
2. **Ontologize**: organize all features into a strict,
   non-redundant `DOMAIN -> SUBDOMAIN -> FEATURE` subclass hierarchy by **meaning**,
   never by statistics (two unrelated measures can correlate by accident). The model
   is given each feature's label, description, units, source and sample values
   (as TOON), proposes the domains itself, then organises each domain into subdomains
   in parallel; code enforces exact, non-redundant coverage. This is general for any
   dataset and scales via per-domain calls. An optional free-text user-guidance
   argument is injected into every prompt (the hook for a future UI). The result is
   quality-assessed (adjusted Rand index against the statistical clusters, plus a
   compact LLM MECE review). Built **once per dataset** and reused as a fixed base
   template for every subject, so participants differ only in which leaf values are
   present, never in structure. Emits Protege-loadable OWL, a subclass JSON, a single
   hierarchy-encoded benchmark CSV, an interactive HTML explorer, and QA reports.
3. **Encode**: standardize each feature into a deviation score with its label
   preserved. Reference strategy is auto-selected: `cohort` (batch is its own
   reference), `external` (supplied norms), or `absolute` (no reference / single
   subject, with an optional LLM range estimate).
4. **Write**: render the four participant input files consumed by COMPASS.

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

| Dataset | Source | Task | Modalities | Status |
|---|---|---|---|---|
| [AOMIC-ID1000](datasets/AOMIC_ID1000/) | OpenNeuro ds003097 (CC0) | Native IST total-score regression | Self-report, FreeSurfer morphometry, fMRI connectome | 874 of 900 valid predictions; 8 tiers complete |

More datasets plug in by copying a dataset folder and editing its `pipeline/config.py`.

## Requirements

`OPENROUTER_API_KEY` in the repository-root `.env`. LLM steps use a small model
(`google/gemini-3.1-flash-lite`) by default. Brain extraction needs `nilearn`,
`nibabel`, and network access to the OpenNeuro S3 mirror.
