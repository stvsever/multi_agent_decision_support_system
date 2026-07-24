# NUMERACY_STROKE validation (OpenNeuro ds006533)

Validation of COMPASS on precise vs. approximate numeracy in 105 left-hemisphere
chronic stroke survivors, from demographics and whole-brain lesion-overlap features.

See [PHENOTYPE_AND_TIERS.md](PHENOTYPE_AND_TIERS.md) for the exact clinical phenotype
prediction structure (two dissociable numeracy phenotypes) and the data-complexity
tier ladder.

## Layout

```text
NUMERACY_STROKE/
  README.md                  this file
  PHENOTYPE_AND_TIERS.md     phenotype output structure + tier ladder
  data/
    raw/ds006533/            lesion masks + participants.tsv (gitignored, re-fetchable)
    processed/               per-subject and _all_subjects feature tables
                             (raw + transformed; only transformed is ingested)
  pipeline/
    01_download_lesion_data.py   fetch lesion masks
    02_build_processed_tables.py raw/transformed per-subject and cohort tables
    03_build_ontology.py         deterministic abstract ontology (fine + coarse)
    04_build_compass_inputs.py   the four COMPASS files per subject per tier
    compass_task.py              task spec, global instruction, engine run helpers
  ontology/                  subclass_structure_{fine,coarse}.json/.owl
  compass_inputs/<tier>/     blinded + all-shared inputs per participant
  results/
    annotations.json         ground truth (both targets) for all 105 subjects
    subset_<target>.json      blinded evaluation ground truth per target
```

## Reproduce

```bash
# raw -> processed tables (needs nibabel + cached atlases; heavy)
python pipeline/02_build_processed_tables.py
# ontology (deterministic, LLM-free; reuses the coarse table if present)
python pipeline/03_build_ontology.py
# the four COMPASS files per subject per tier
python pipeline/04_build_compass_inputs.py
```

Inference is driven from the shared validation notebook
(`validation/datasets/validation_with_openneuro_datasets.ipynb`), which imports
`pipeline/compass_task.py` for the task spec and global instruction. Only the
standardized TRANSFORMED feature tier is ingested; the engine is told the scales of
every feature and target in the global instruction and in each record's SCALE GUIDE.
Requires `OPENROUTER_API_KEY` in the repository-root `.env`.
