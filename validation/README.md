# COMPASS cross-dataset validation

This directory validates the COMPASS engine against real, open-access datasets whose
structure differs from the UK Biobank layout the engine was originally shaped around.
It demonstrates the engine's **flexible ingestion**: any pre-processed multi-modal
dataset is projected onto the engine's contract without touching `src/full_stack`.

**Design rule:** all dataset-specific code, data, and results live here. The
full-stack engine stays dataset-agnostic.

## Structure

```
validation/
  common/                      reusable, dataset-agnostic ingestion engine
    explore.py                   automated data understanding (types, correlations,
                                 feature clustering, redundancy, quality flags)
    manifest.py                  deterministic feature-structure profiling
    ontology.py                  LLM ontology (hint or auto) + QA review + OWL/JSON
    deviation.py                 reference strategies (cohort / external / absolute)
    compass_writer.py            emits the four COMPASS participant files
    freesurfer.py                FreeSurfer morphometry feature extraction
    connectome.py                movie-fMRI network connectome extraction
    tiers.py                     project the ontology onto a complexity tier
    llm.py                       minimal OpenRouter client (reads repo-root .env)
  datasets/
    AOMIC_ID1000/              first validated dataset (OpenNeuro ds003097)
      dataset/                   raw CC0 data + data dictionary
      pipeline/                  config + numbered sequential scripts (01..05, 10, 11)
      brain/                     FreeSurfer + connectome extracted features
      ontology/                  master ontology (OWL + subclass JSON) + manifest
      compass_inputs/<tier>/     generated engine inputs per tier and participant
      results/<tier>/            predictions, metrics, full engine outputs per tier
      notebooks/                 executed preprocessing + visualisation notebooks
```

## How ingestion works

1. **Explore** - a sophisticated automated pass that *understands* any tabular folder:
   automatic type inference, distribution and missingness profiling, robust
   correlation structure, hierarchical feature clustering, near-duplicate detection,
   target associations, and quality flags. No hand-written hints required.
2. **Ontologise** - organise all features into a strict, non-redundant
   `DOMAIN -> SUBDOMAIN -> FEATURE` subclass hierarchy. Structure comes from
   per-feature hints when domain knowledge exists, or from the exploration's
   data-driven clusters otherwise; a small model generates only the interpretable
   parent labels and definitions (leaves keep labels, no redundant descriptions).
   The prompt is serialised as TOON, not JSON, to save tokens. The result is then
   quality-assessed: adjusted Rand index against the statistical clusters plus a
   compact LLM MECE/coherence review. The ontology is built **once per dataset** and
   reused as a fixed base template for every subject, so participants differ only in
   which leaf values are present, never in structure. Emits Protege-loadable OWL, a
   subclass JSON, and exploration/quality reports.
3. **Encode** - standardise each feature into a deviation score with its label
   preserved. Reference strategy is auto-selected: `cohort` (batch is its own
   reference), `external` (supplied norms), or `absolute` (no reference / single
   subject, with an optional LLM range estimate).
4. **Write** - render the four engine files.

The engine ingests those files exactly as it does UK-Biobank-style inputs and, like
the engine's own agents, reasons over the token-efficient TOON representation of the
data. Because feature **labels** travel with every value, the multi-agent system can
interpret what each number means.

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
| [AOMIC-ID1000](datasets/AOMIC_ID1000/) | OpenNeuro ds003097 (CC0) | Total-IQ univariate regression | Self-report, FreeSurfer morphometry, fMRI connectome | Validated across 9 complexity tiers |

More datasets plug in by copying a dataset folder and editing its `pipeline/config.py`.

## Requirements

`OPENROUTER_API_KEY` in the repository-root `.env`. LLM steps use a small model
(`google/gemini-3.1-flash-lite`) by default. Brain extraction needs `nilearn`,
`nibabel`, and network access to the OpenNeuro S3 mirror.
