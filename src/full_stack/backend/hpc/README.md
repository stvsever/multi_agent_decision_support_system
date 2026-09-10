# COMPASS HPC Scripts (Slurm + Apptainer, single GPU to multi node)

This folder contains an **example HPC workflow** for running COMPASS on Slurm-based GPU clusters.

It was developed to support **phenotype validation runs** of the COMPASS engine on a **single-GPU node** (example target hardware: **NVIDIA L40S, 48 GB VRAM**). The defaults in these scripts reflect that operating constraint.

It is intentionally written as a reproducible template for:
- Single-participant validation (`04_submit_single.sh`)
- Sequential batch validation (`05_submit_batch.sh`)
- Serving open weights to any number of COMPASS clients (`06_serve_vllm.sh`)

The single-GPU defaults are still the defaults, but nothing is nailed to one GPU
any more: every Slurm resource and both parallel sizes are environment variables.

Use it as a starting point. For your cluster, adapt partition/GPU/path settings as needed.

## Scope and assumptions

- Scheduler: Slurm
- Container runtime: Apptainer
- Execution mode: single GPU and sequential participant processing by default (stability-first); multi-GPU and multi-node are opt-in, see below
- Main local model: `Qwen/Qwen3-14B-AWQ`
- Embedding model: `Qwen/Qwen3-Embedding-8B`
- Project path on cluster: `~/compass_pipeline/multi_agent_decision_support_system`

## What these scripts do

- `00_deploy_and_run.sh`: optional helper to copy this repo (and optionally data) to an HPC and SSH in
- `01_check_status.sh`: pre-flight checks (paths, models, Slurm/apptainer availability)
- `02_setup_environment.sh`: create container + venv environment
- `03_download_models.sh`: download/patch models in shared storage
- `04_submit_single.sh`: submit one participant smoke test job
- `05_submit_batch.sh`: submit sequential batch run across participant list in `src/full_stack/backend/utils/batch_run.py`
- `06_serve_vllm.sh`: start a vLLM server for a HuggingFace model and print the base URL to point COMPASS at
- `apptainer.def`: definition for a container carrying CUDA, vLLM, and the engine dependencies
- `HPC_Operational_Guide.ipynb`: didactic notebook explaining HPC components and the end-to-end workflow with these scripts

## Notebook-first onboarding (recommended)

If you are new to HPC execution, start with:

- `src/full_stack/backend/hpc/HPC_Operational_Guide.ipynb`

It explains Slurm, login-vs-compute nodes, Apptainer, runtime tradeoffs, monitoring, and migration to other clusters.

## Important: these defaults are examples

These scripts currently include concrete defaults such as:
- `#SBATCH --partition=main`
- `#SBATCH --gres=gpu:l40s:1`
- `PROJECT_DIR="${HOME}/compass_pipeline/multi_agent_decision_support_system"`

Treat these as sample values and update them for your infrastructure.

Design choices you will likely adapt for other hardware:
- GPU request (`--gres`) and partition/queue names
- memory/time requests (`--mem`, `--time`)
- context and per-role token budgets (longer context and larger outputs increase latency and memory pressure)
- whether to remain sequential (single GPU) or introduce safe parallelism (multiple GPUs or multiple nodes)

## Recommended run order

From `~/compass_pipeline/multi_agent_decision_support_system`:

```bash
bash src/full_stack/backend/hpc/00_deploy_and_run.sh   # optional
bash src/full_stack/backend/hpc/01_check_status.sh
bash src/full_stack/backend/hpc/02_setup_environment.sh
bash src/full_stack/backend/hpc/03_download_models.sh
bash src/full_stack/backend/hpc/04_submit_single.sh
bash src/full_stack/backend/hpc/05_submit_batch.sh
bash src/full_stack/backend/hpc/06_serve_vllm.sh   # optional, server mode
```

All submission scripts are login-node safe:
- If run on `login*`, they auto-submit to Slurm compute nodes.
- Logs are written to `~/compass_pipeline/multi_agent_decision_support_system/logs/`.

## Monitoring commands

```bash
squeue -u "$USER"
tail -f logs/compass_single_<JOBID>.out
tail -f logs/compass_batch_<JOBID>.out
tail -f logs/vllm_serve_<JOBID>.out
cat logs/compass_single_<JOBID>.err
cat logs/compass_batch_<JOBID>.err
cat logs/vllm_endpoint_<JOBID>.txt
```

## Two ways to use the GPUs

**In-process (steps 04 and 05).** COMPASS loads the model inside its own process
with vLLM or Transformers. One job owns one model for its whole lifetime. This is
the validation path, and it stays the default.

**Server mode (step 06).** `06_serve_vllm.sh` starts a vLLM server once. Any
number of COMPASS runs then talk to it over the OpenAI protocol: other jobs on
the cluster, or the dashboard on your laptop through an SSH tunnel. Use it when
several runs or several people share one model, when you want to keep the model
warm between runs, or when a model is too large for one node.

## Multi-GPU and multi-node

Every Slurm resource in steps 04, 05, and 06 is an environment variable, so a
larger allocation needs no edit to any script. The `#SBATCH` lines inside each
script remain the defaults for a direct `sbatch`.

What the extra hardware buys you is not the same in every step. Steps 04 and 05
run a single `apptainer exec` on the first node of the allocation: no `srun`
fan-out, no Ray cluster. A model there is confined to the GPUs of that one node
however many nodes the job holds, so raising `COMPASS_SLURM_NODES` for them only
idles the extra machines. Step 06 is the multi-node path: it starts a Ray head
and one worker per extra node, then serves the model across all of them.

| Variable | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `COMPASS_SLURM_PARTITION` | `main` | 04, 05, 06 | queue to submit to |
| `COMPASS_SLURM_NODES` | `1` | 06 only | nodes in the allocation |
| `COMPASS_SLURM_NTASKS_PER_NODE` | `1` | 04, 05, 06 | tasks per node |
| `COMPASS_SLURM_CPUS_PER_TASK` | `16` | 04, 05, 06 | CPU cores per task |
| `COMPASS_SLURM_MEM` | `64G` | 04, 05, 06 | host memory per node |
| `COMPASS_SLURM_GRES` | `gpu:l40s:1` | 04, 05, 06 | GPUs per node, for example `gpu:l40s:4` |
| `COMPASS_SLURM_TIME` | per script | 04, 05, 06 | wall clock limit |
| `LOCAL_TENSOR_PARALLEL` | `1` | 04, 05 | GPUs one model is split across, on this node |
| `LOCAL_PIPELINE_PARALLEL` | `1` | 04, 05 | pipeline stages, also on this node |
| `TP_SIZE` | `auto` | 06 | GPUs one model is split across, within a node |
| `PP_SIZE` | `auto` | 06 | pipeline stages, across the nodes Ray joins |

In steps 04 and 05 both parallel sizes are spent on one node, so their product
must not exceed the GPUs that node has: `LOCAL_TENSOR_PARALLEL=2` with
`LOCAL_PIPELINE_PARALLEL=2` asks for four GPUs, exactly like
`LOCAL_TENSOR_PARALLEL=4` does. Outside Slurm the scripts fall back to
`CUDA_VISIBLE_DEVICES=0,...,(TP * PP - 1)` for the same reason.

Tensor parallel size must divide the model's attention head count, and must not
exceed the GPUs you asked for. A four-GPU single-participant run:

```bash
COMPASS_SLURM_GRES=gpu:l40s:4 \
COMPASS_SLURM_CPUS_PER_TASK=32 \
COMPASS_SLURM_MEM=256G \
LOCAL_TENSOR_PARALLEL=4 \
  bash src/full_stack/backend/hpc/04_submit_single.sh
```

The same variables work for `05_submit_batch.sh`. Batch execution stays
sequential per participant; the extra GPUs make each participant faster rather
than running participants in parallel. For a model that does not fit on one
node, serve it with step 06 and point the run at that server rather than loading
it in process.

## Serving a model with vLLM (step 06)

```bash
# 1. One GPU
bash src/full_stack/backend/hpc/06_serve_vllm.sh

# 2. Four GPUs on one node, tensor parallel
COMPASS_SLURM_GRES=gpu:l40s:4 TP_SIZE=4 \
  bash src/full_stack/backend/hpc/06_serve_vllm.sh

# 3. Two nodes with four GPUs each: tensor parallel inside a node,
#    pipeline parallel across nodes, wired together by Ray
COMPASS_SLURM_NODES=2 COMPASS_SLURM_GRES=gpu:l40s:4 TP_SIZE=4 PP_SIZE=2 \
  bash src/full_stack/backend/hpc/06_serve_vllm.sh

# 4. A different model
MODEL_ID=Qwen/Qwen3-32B-AWQ bash src/full_stack/backend/hpc/06_serve_vllm.sh

# 5. On a workstation with Docker and no scheduler
RUNTIME=docker bash src/full_stack/backend/hpc/06_serve_vllm.sh
```

Run `bash src/full_stack/backend/hpc/06_serve_vllm.sh --help` for the full list
of settings. `TP_SIZE` and `PP_SIZE` default to `auto`, which reads the GPUs and
nodes the scheduler actually gave the job. `MODEL_ID` may be a HuggingFace repo
id or a local path; a repo already downloaded by step 03 is preferred over the
network, so the server works on nodes without outbound internet. That resolved
path is what every runtime is handed, including `RUNTIME=docker`, which binds
`MODELS_DIR` into the container so the local copy is the one that loads.

Which Python environment the server command runs in follows the image, not the
host, and the job log prints the choice as `Python env:`. An image that carries
vLLM at `/opt/venv` (`compass_gpu.sif`, or `compass-engine:gpu` under Docker) is
used as it is; the NGC PyTorch container from step 02 has no vLLM of its own, so
the `~/compass_venv` venv is sourced on top of it. Sourcing that venv inside
`compass_gpu.sif` would shadow the image's own interpreter, which is why the
choice is not simply "the venv exists". Force the issue with
`COMPASS_SERVE_VENV=none` or `COMPASS_SERVE_VENV=force`.

`TRUST_REMOTE_CODE` defaults to `1`, so a repository that ships its own modeling
code loads. Set it to `0` to refuse. The same default and the same off switch
exist as `COMPASS_VLLM_TRUST_REMOTE_CODE` in `apptainer.def` and in
`docker/entrypoint.gpu.sh`, so all three runtimes serve identical vLLM flags.

When the server answers its health probe, the job log prints the base URL and
writes it to `logs/vllm_endpoint_<JOBID>.txt`:

```text
base_url=http://gpu042.cluster.local:8000/v1
model=Qwen/Qwen3-14B-AWQ
api_key=local-vllm
```

Point COMPASS at it. The engine speaks the OpenAI protocol, so a self-hosted
server is configured the same way as a hosted provider:

```bash
export OPENROUTER_BASE_URL="http://gpu042.cluster.local:8000/v1"
export OPENROUTER_API_KEY="local-vllm"
python3 main.py <participant_dir> --backend openrouter --model "Qwen/Qwen3-14B-AWQ" ...
```

From a laptop, tunnel through the login node first, then use
`http://localhost:8000/v1` in the dashboard under Settings, Connection:

```bash
ssh -N -L 8000:gpu042.cluster.local:8000 <user>@<cluster-login-host>
```

Stop the server with `scancel <JOBID>`.

## Container image (`apptainer.def`)

`apptainer.def` builds the cluster twin of `docker/Dockerfile.gpu`: CUDA, vLLM,
Ray, and the engine dependencies, with the same two roles (`serve` and
`dashboard`). Steps 04 and 05 keep working with the NGC PyTorch container plus
the venv from step 02; the definition file is there for clusters that want one
self-contained image.

```bash
# from ~/compass_pipeline/multi_agent_decision_support_system
apptainer build ~/compass_containers/compass_gpu.sif \
    src/full_stack/backend/hpc/apptainer.def

# clusters that forbid unprivileged builds: build remotely
apptainer build --remote ~/compass_containers/compass_gpu.sif \
    src/full_stack/backend/hpc/apptainer.def

# or build the Docker image where you have root and convert it
docker build -f docker/Dockerfile.gpu -t compass-engine:gpu .
apptainer build compass_gpu.sif docker-daemon://compass-engine:gpu

# use it with step 06
CONTAINER_IMAGE=~/compass_containers/compass_gpu.sif \
  bash src/full_stack/backend/hpc/06_serve_vllm.sh
```

Two things to remember when running it by hand:

- `--nv` is what exposes the host NVIDIA driver to the container. Without it the
  container starts, sees no GPU, and vLLM fails at load time.
- Apptainer mounts `$HOME` and the current directory by default. Anything else,
  including project or scratch filesystems holding weights or participant data,
  needs an explicit `--bind /path:/path`.

```bash
apptainer run --nv \
    --bind "${HOME}/compass_models:${HOME}/compass_models" \
    --env COMPASS_VLLM_MODEL=Qwen/Qwen3-14B-AWQ \
    --env HF_HOME="${HOME}/compass_models/hf_cache" \
    ~/compass_containers/compass_gpu.sif serve
```

## Single run vs batch run

### `04_submit_single.sh`

- Validates one participant end-to-end.
- Uses local backend settings tuned for a single-GPU run.
- Intended as the gate before batch execution.

### `05_submit_batch.sh`

- Runs the participant cohort defined in `src/full_stack/backend/utils/batch_run.py`.
- Keeps execution **sequential** (single GPU) by design.
- Passes local runtime, token budgets, and prediction task flags through to `main.py` per participant:
  - `PREDICTION_TYPE` (default `binary`)
  - `TARGETS_FILE` must be JSON for binary queue construction (`binary_targets.json` style)
  - Optional `CLASS_LABELS`, `REGRESSION_OUTPUT` (univariate), `REGRESSION_OUTPUTS` (multivariate), `TASK_SPEC_FILE`, `TASK_SPEC_JSON`
  - Optional `ANNOTATIONS_JSON` for non-binary post-hoc validation (`run_validation_metrics.py` / `detailed_analysis.py`)
  - Optional runtime guidance: `GLOBAL_INSTRUCTION`, `ORCHESTRATOR_INSTRUCTION`, `EXECUTOR_INSTRUCTION`, `TOOLS_INSTRUCTION`, `INTEGRATOR_INSTRUCTION`, `PREDICTOR_INSTRUCTION`, `CRITIC_INSTRUCTION`, `COMMUNICATOR_INSTRUCTION`
  - `--max_tokens`
  - `--max_agent_input`, `--max_agent_output`
  - `--max_tool_input`, `--max_tool_output`
  - local engine/quantization settings

### `04_submit_single.sh`

- Uses the same prediction-task controls as Step 05.
- Defaults remain binary (`PREDICTION_TYPE=binary`) so existing case/control workflows still run unchanged.
- You can override to multiclass/regression/hierarchical by exporting:
  - `PREDICTION_TYPE`
  - optional `CLASS_LABELS`, `REGRESSION_OUTPUT` (univariate), `REGRESSION_OUTPUTS` (multivariate), `TASK_SPEC_FILE`, `TASK_SPEC_JSON`

For non-binary post-hoc validation in Step 05, also export:

```bash
ANNOTATIONS_JSON=/path/to/annotated_targets.json
```

Post-hoc analysis output directories are mode-aware:
- binary: `results/analysis/binary_confusion_matrix/` and `results/analysis/details/`
- non-binary: `results/analysis/<prediction_type>_metrics/` and `results/analysis/<prediction_type>_details/`

Non-binary detailed outputs include:
- annotation contract reports (`detailed_annotation_contract_<prediction_type>.json/.txt`)
- row-level evaluation payloads (`detailed_rows_<prediction_type>.json`)
- mode-specific plots (for example multiclass top-confusions, regression residual diagnostics, hierarchical coverage)

## Participant cohort definition

`src/full_stack/backend/utils/batch_run.py` contains the participant subset (EIDs + expected label + target string):

- Edit `PARTICIPANTS` in `src/full_stack/backend/utils/batch_run.py` to change the batch cohort.
- Data folder is resolved from:
  - `DATA_ROOT` env var if set, else
  - `../data/__FEATURES__/HPC_data`

Expected folder pattern:

```text
.../HPC_data/participant_ID<eid>/
```

## Local backend and public API backend behavior

The pipeline supports both:
- Local model inference (HPC/local GPU)
- Public API inference (OpenRouter/OpenAI)

They are intentionally configurable independently. The HPC scripts in this folder are focused on the **local backend** path for clinical validation runs.

Note: Explainability (XAI) remains binary-root-only. Non-binary prediction modes run normally, but XAI steps are skipped with explicit status metadata.

## Performance notes

- Local open-source inference on 1x L40S is expected to be slower than hosted public APIs.
- Most runtime is model generation latency and repeated agent/tool calls (COMPASS is intentionally multi-step).
- Longer context windows and higher output budgets increase latency.
- `05_submit_batch.sh` is tuned for reliability on single-GPU sequential execution.

## Quick adaptation checklist for other clusters

1. Set the Slurm request through `COMPASS_SLURM_*`, or update the `#SBATCH`
   defaults in `04_submit_single.sh`, `05_submit_batch.sh`, and `06_serve_vllm.sh`.
2. Update paths (`PROJECT_DIR`, `MODELS_DIR`, `VENV_DIR`, `CONTAINER_IMAGE`).
3. Confirm `apptainer` command availability on compute nodes.
4. Run `01` -> `04` successfully before running `05`.
5. For server mode, check that compute nodes are reachable on the serving port,
   or tunnel through the login node.
