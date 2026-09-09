<div align="center">

# Clinical Ontology-driven Multi-modal Predictive Agentic Support System (COMPASS)

[![Software Tool](https://img.shields.io/badge/Type-Software_Tool-4f46e5.svg?style=flat-square)](#)
[![License: GPL 3.0](https://img.shields.io/badge/License-GPL_3.0-059669.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-0db7ed.svg?style=flat-square&logo=docker&logoColor=white)](docker/)
<br>

**COMPASS** is a flexible multi-agent decision support engine for deep phenotype prediction. It combines hierarchical multi-modal deviation maps, structured feature data, and non-tabular health information through an orchestrated actor-critic workflow. The engine supports binary classification, multiclass classification, univariate regression, multivariate regression, and hierarchical mixed task trees.

</div>

---

## 📖 Table of Contents

- [🚀 Key Features](#key-features)
- [🧠 System Architecture](#system-architecture)
- [🖥️ Interactive Dashboard](#interactive-dashboard)
- [🛠️ Installation](#installation)
- [⚡ Usage](#usage)
- [📁 Project Structure](#project-structure)
- [📈 Development Status](#development-status)
- [👥 Team](#team)

<br>

## <a id="key-features"></a>🚀 Key Features

- **Multi-Agent Orchestration**: A dynamic actor-critic workflow coordinates the Orchestrator, Executor, Integrator, Predictor, Critic, and Communicator agents.
- **Flexible Prediction Tasks**: Typed task specifications support classification, regression, and mixed hierarchical output trees.
- **No-Loss Evidence Flow**: Feature-level coverage tracking preserves processed and unprocessed multi-modal evidence through integration and chunking.
- **Explainable Clinical Reasoning**: Optional XAI methods and evidence chains connect predictions to source features and clinical narratives.
- **Live Dashboard**: The web interface exposes execution plans, agent progress, token usage, predictions, critic feedback, and generated reports.
- **Deep Phenotyping Reports**: The Communicator agent generates an evidence-grounded `deep_phenotype.md` report and marks missing information explicitly.

## <a id="system-architecture"></a>🧠 System Architecture

![COMPASS multi-agent workflow](report/objects/figures/main/MAIN_01_flowchart.png)

The Executor can run independent tool steps concurrently for public API backends. Local inference uses sequential execution to reduce GPU memory pressure. After the final iteration, COMPASS selects the strongest satisfactory attempt. If no attempt is satisfactory, it selects the highest-scoring attempt and records that status in the report.

## <a id="interactive-dashboard"></a>🖥️ Interactive Dashboard

A FastAPI service supervises runs as isolated worker processes and serves a React web client from the same port.

The dashboard supports:

- guided setup for the provider key, model, and data folders;
- binary, multiclass, regression, and hierarchical task design;
- cost projection from live OpenRouter pricing, with warning and hard-stop thresholds;
- live execution streamed over server-sent events, with the orchestrator plan rendered as an animated graph of sequential, parallel, and feedback edges;
- an ontology explorer over the hierarchical deviation map and feature payload, as a tree, sunburst, icicle, or ranked table;
- prediction, critic checklist, token, and per-model cost inspection;
- deep phenotype reports on screen and as a standardized PDF;
- batch execution across a cohort with configurable concurrency;
- high-resolution control over per-role models, token ceilings, temperatures, reasoning effort, executor workers, budgets, agent instructions, and the system prompts themselves.

Build the web client once, then launch:

```bash
npm --prefix src/full_stack/frontend install
npm --prefix src/full_stack/frontend run build
python3 main.py --ui
```

The dashboard is served at http://127.0.0.1:5005. Set `COMPASS_UI_HOST` and `COMPASS_UI_PORT` to change that.

For web client development, run the API and the Vite dev server side by side:

```bash
python3 main.py --ui --quiet
npm --prefix src/full_stack/frontend run dev
```

Vite serves the client on http://localhost:5173 and proxies `/api` to the service.

The bundled pseudo-participant folders are discovered automatically and can be run directly from the dashboard.

## <a id="installation"></a>🛠️ Installation

### Local development

Python 3.11 or newer runs the engine. Node 20 or newer builds the dashboard client.

```bash
git clone https://github.com/stvsever/multi_agent_decision_support_system.git
cd multi_agent_decision_support_system
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env, or paste it into the dashboard settings
```

OpenRouter is the default public backend and `deepseek/deepseek-v4-flash-0731` is the default model for all agent and tool roles. Models, token ceilings, temperatures, and reasoning effort remain configurable through CLI flags and the dashboard.

Reasoning is off by default. Reasoning tokens are billed as output and count against the output ceiling, so on a reasoning model they can consume the whole budget and truncate the answer. Raise it with `--reasoning_effort` or in the dashboard when a task needs deeper deliberation.

The local `.env` file is loaded automatically and is excluded from Git. Model, schema, or connectivity failures stop the run explicitly. COMPASS does not replace failed LLM outputs with deterministic predictions, plans, evaluations, or narratives.

### Docker (CPU/UI)

For the complete container workflow, see [docker/README.md](docker/README.md).

```bash
tar --exclude-from=docker/.dockerignore -cf - . | docker buildx build \
  --platform linux/arm64 \
  -f docker/Dockerfile \
  -t compass-ui:local \
  --load \
  -

docker run --rm \
  -p 5005:5005 \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  --name compass-ui \
  compass-ui:local
```

Use `--platform linux/amd64` on Intel Mac, Linux, and Windows Docker Desktop.

> [!NOTE]
> The default Docker image is CPU-first and uses public API inference. The optional full image includes local inference dependencies. GPU and Slurm workflows remain under `src/full_stack/backend/hpc/`.

## <a id="usage"></a>⚡ Usage

### Expected input and output structure

Each participant folder must contain:

```text
data_overview.json
hierarchical_deviation_map.json
multimodal_data.json
non_numerical_data.txt
```

This repository currently includes synthetic pseudo-data for development and testing under:

```text
src/full_stack/backend/data/pseudo_data/inputs/
```

Generated pseudo-data outputs are written under `src/full_stack/backend/data/pseudo_data/outputs/` and are ignored by Git. Other run outputs are written to `results/`.

### Backend smoke test without LLM calls

The offline audit validates loading, predictor payload construction, feature coverage, and chunking:

```bash
python3 main.py \
  src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO \
  --prediction_type binary \
  --target_label target_phenotype \
  --control_label non_target_comparator \
  --audit
```

### Full CLI run

```bash
python3 main.py \
  src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO \
  --prediction_type binary \
  --target_label target_phenotype \
  --control_label non_target_comparator \
  --backend openrouter \
  --public_model deepseek/deepseek-v4-flash-0731
```

Other task modes:

```bash
# Multiclass
python3 main.py src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO \
  --prediction_type multiclass \
  --target_label phenotype_subtype \
  --class_labels subtype_a,subtype_b,subtype_c

# Univariate regression
python3 main.py src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO \
  --prediction_type regression_univariate \
  --target_label total_score \
  --regression_output total_score

# Multivariate regression
python3 main.py src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO \
  --prediction_type regression_multivariate \
  --target_label phenotype_profile \
  --regression_outputs phenotype_p1,phenotype_p2,phenotype_p3

# Hierarchical mixed task tree
python3 main.py src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO \
  --prediction_type hierarchical \
  --task_spec_file /path/to/task_spec.json
```

### Explainability

```bash
python3 main.py src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO \
  --prediction_type binary \
  --target_label target_phenotype \
  --control_label non_target_comparator \
  --backend openrouter \
  --xai_methods external,internal,hybrid
```

XAI currently supports pure root-level binary classification. Other task modes run normally and record an explicit XAI skip status.

### Clinical validation

The annotated validation utilities are located under `src/full_stack/backend/utils/validation/with_annotated_dataset/`.

```bash
python3 src/full_stack/backend/utils/validation/with_annotated_dataset/run_validation_metrics.py \
  --results_dir results/participant_runs \
  --prediction_type binary \
  --targets_file /path/to/binary_targets.json \
  --output_dir results/analysis/binary_confusion_matrix
```

See [validation_guide.ipynb](src/full_stack/backend/utils/validation/with_annotated_dataset/validation_guide.ipynb) for the complete validation workflow.

### HPC example

The Slurm and Apptainer templates remain available under `src/full_stack/backend/hpc/`. They were updated for the new repository layout but are not part of the local automated test run.

See [HPC README](src/full_stack/backend/hpc/README.md) and [HPC Operational Guide](src/full_stack/backend/hpc/HPC_Operational_Guide.ipynb).

## <a id="project-structure"></a>📁 Project Structure

```text
multi_agent_decision_support_system/
├── docker/                         # CPU/UI and full container definitions
├── report/                         # Manuscript sources and generated report objects
├── src/
│   ├── full_stack/
│   │   ├── backend/
│   │   │   ├── agents/            # Agent implementations and prompts
│   │   │   ├── api/               # FastAPI dashboard service, run supervision, cost, PDF
│   │   │   ├── config/            # Runtime and model configuration
│   │   │   ├── data/              # Typed models and pseudo-data
│   │   │   ├── hpc/               # Slurm and Apptainer templates
│   │   │   ├── runtime/           # Engine event bus consumed by the dashboard
│   │   │   ├── tools/             # Clinical analysis tools and prompts
│   │   │   └── utils/             # Core engine, validation, XAI, and logging
│   │   └── frontend/               # React and Vite web client
│   └── tests/                      # Engine and dashboard unit tests
├── validation/
│   ├── validation_with_openneuro_datasets.ipynb  # Master notebook: runs all 3 datasets end to end (resumable)
│   ├── common/                     # Reusable exploration, ontology, ingestion, MRI, and evaluation code
│   └── datasets/
│       ├── INTELLIGENCE/           # OpenNeuro ds003097 (AOMIC-ID1000): hierarchical IQ inference
│       │   ├── brain/              # Derived high-resolution morphometry and connectome features
│       │   ├── compass_inputs/     # Blinded inputs per tier and participant
│       │   ├── dataset/            # Source participant table and field metadata
│       │   ├── ontology/           # Generated arbitrary-depth ontology and reports
│       │   ├── pipeline/           # Extraction, inference, and evaluation scripts
│       │   ├── results/            # Per-tier predictions/metrics + annotations.json
│       │   ├── PHENOTYPE_AND_TIERS.md  # Phenotype output structure and tier ladder
│       │   ├── METHODOLOGY.md       # Leakage controls and evaluation protocol
│       │   └── README.md
│       ├── PSYCHOSIS_FIRST_EPISODE/  # OpenNeuro ds003944 + ds003947: diagnosis + symptom profile from EEG
│       │   ├── data/               # Raw/processed EEG (git-ignored) + derived feature tables
│       │   ├── notebooks/          # Load/preprocess, feature extraction/viz, ontology + COMPASS ladder
│       │   ├── utils/              # Importable pipeline: features, viz, ontology, task spec, run helpers
│       │   ├── results/compass/    # Ontology (OWL), per-tier inputs, annotations, ladder predictions
│       │   └── PHENOTYPE_AND_TIERS.md
│       └── NUMERACY_STROKE/        # OpenNeuro ds006533: approximate vs precise numeracy from lesion overlap
│           ├── data/              # Lesion masks (git-ignored) + processed feature tables
│           ├── ontology/          # Fine/coarse abstract lesion ontology (OWL + JSON)
│           ├── pipeline/          # Lesion extraction, ontology, COMPASS inputs, task spec
│           ├── compass_inputs/    # Blinded + all-shared inputs per tier and participant
│           ├── results/           # subset + annotations.json ground truth
│           ├── PHENOTYPE_AND_TIERS.md
│           └── README.md
├── COMPASS_demo.ipynb              # End-to-end demonstration notebook
├── main.py                         # CLI and UI entry point
├── pyproject.toml                  # Package metadata and build configuration
├── requirements.txt               # Python dependencies
├── CITATION.cff                    # Citation metadata
├── LICENSE
└── README.md
```

## <a id="development-status"></a>📈 Development Status

COMPASS is an active research prototype. The actor-critic pipeline, generalized task contracts, no-loss evidence coverage, dashboard, Docker runtime, validation utilities, and pseudo-data workflows are functional. Ongoing work focuses on stability, calibration, testing, reporting, and scalable research use.

Run the automated test suite with:

```bash
python3 -m pytest -q
```

## <a id="team"></a>👥 Team

- Stijn Van Severen¹
- Gavin Schneider²
- Ziyuan Chen²

*¹ Department of Experimental Psychology, Ghent University, Ghent, Belgium*
*² Department of Psychology, University of Oregon, Eugene, OR*

> [!CAUTION]
> **PRE-CLINICAL DISCLAIMER**
> COMPASS is a research prototype and is not a certified medical device under the EU Medical Device Regulation or FDA requirements. Do not use it for primary diagnostic decisions. All outputs require review by qualified domain experts.
