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

The dashboard supports:

- live execution and stage monitoring;
- execution plan inspection;
- role-specific model and token configuration;
- binary, multiclass, regression, and hierarchical task setup;
- prediction and critic inspection;
- structured input, output, and report browsing.

Launch it with:

```bash
python3 main.py --ui
```

The bundled pseudo-participant folders are discovered automatically and can be launched directly from the UI.

## <a id="installation"></a>🛠️ Installation

### Local development

```bash
git clone https://github.com/stvsever/multi_agent_decision_support_system.git
cd multi_agent_decision_support_system
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env
```

OpenRouter is the default public backend. The testing profile uses `google/gemini-3.1-flash-lite` for all agent and tool roles. Models remain configurable through CLI flags and the dashboard.

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
  --public_model google/gemini-3.1-flash-lite
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
├── docker/                         # CPU/UI and full container images
├── report/                         # Local LaTeX reporting workspace, ignored by Git
├── src/
│   ├── full_stack/
│   │   ├── backend/
│   │   │   ├── agents/            # Agent implementations and prompts
│   │   │   ├── config/            # Runtime and model configuration
│   │   │   ├── data/              # Typed models and pseudo-data
│   │   │   ├── hpc/               # Slurm and Apptainer templates
│   │   │   ├── tools/             # Clinical analysis tools and prompts
│   │   │   └── utils/             # Core engine, validation, XAI, and logging
│   │   └── frontend/               # Flask dashboard, templates, and static assets
│   └── tests/                      # Backend and frontend unit tests
├── COMPASS_demo.ipynb
├── main.py                         # CLI and UI entry point
├── requirements.txt
└── README.md
```

## <a id="development-status"></a>📈 Development Status

COMPASS is an active research prototype. The actor-critic pipeline, generalized task contracts, no-loss evidence coverage, dashboard, Docker runtime, validation utilities, and pseudo-data workflows are functional. Ongoing work focuses on stability, calibration, testing, reporting, and scalable research use.

Run the automated test suite with:

```bash
python3 -m pytest -q
```

> [!CAUTION]
> **PRE-CLINICAL DISCLAIMER**
> COMPASS is a research prototype and is not a certified medical device under the EU Medical Device Regulation or FDA requirements. Do not use it for primary diagnostic decisions. All outputs require review by qualified domain experts.
