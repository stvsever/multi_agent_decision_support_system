# COMPASS cross-dataset validation

This directory validates the COMPASS engine against real, open-access datasets
whose structure differs from the UK Biobank layout the engine was originally
shaped around. It demonstrates the engine's **flexible ingestion**: any
pre-processed multi-modal dataset can be projected onto the engine's contract
without touching `src/full_stack`.

**Design rule:** all dataset-specific code, data, and results live here in
`validation/`. The full-stack engine stays dataset-agnostic.

## Layout

```
validation/
  common/          reusable, dataset-agnostic ingestion helpers
    manifest.py        deterministic feature-structure profiling
    ontology.py        LLM-based non-redundant subclass ontology + OWL/JSON writers
    deviation.py       reference strategies (cohort / external / absolute) + encoding
    compass_writer.py  emits the four COMPASS participant files
    llm.py             minimal OpenRouter client (reads repo-root .env)
  AOMIC_ID1000/    first validated dataset (OpenNeuro ds003097, total-IQ regression)
```

## Ingestion in four stages

1. **Explore** (hardcoded) - profile every candidate feature: statistical type,
   cardinality, missingness, distribution. No LLM.
2. **Ontologise** (LLM) - organise all features into a strict, non-redundant
   `DOMAIN -> SUBDOMAIN -> FEATURE` subclass hierarchy. The proposal is validated
   and repaired in code so every feature is covered exactly once. Emits a
   Protege-loadable OWL file and a subclass-structure JSON.
3. **Encode** - standardise each feature into a deviation score with a
   human-readable label preserved. Reference strategy is explicit and
   auto-selected:
   - `cohort` - the batch is its own reference (needs enough participants),
   - `external` - supplied normative statistics,
   - `absolute` - no reference available (single subject or no reference group):
     raw values pass through, with an optional LLM range estimate for
     High/Normal/Low context.
4. **Write** - render the four engine files (`data_overview.json`,
   `hierarchical_deviation_map.json`, `multimodal_data.json`,
   `non_numerical_data.txt`).

The engine then ingests those files exactly as it does UK-Biobank-style inputs.
Because feature **labels** are carried alongside every value, the multi-agent
system can interpret what each number means.

## Datasets

| Dataset | Source | Task | Status |
|---|---|---|---|
| [AOMIC-ID1000](AOMIC_ID1000/) | OpenNeuro ds003097 (CC0) | Total-IQ univariate regression | Validated (subset) |

More datasets (other OpenNeuro studies, HCP connectome, and further modalities)
plug in by copying a dataset folder and editing its `pipeline/config.py`.

## Requirements

`OPENROUTER_API_KEY` in the repository-root `.env`. Steps that call an LLM use a
small model (`google/gemini-3.1-flash-lite`) by default.
