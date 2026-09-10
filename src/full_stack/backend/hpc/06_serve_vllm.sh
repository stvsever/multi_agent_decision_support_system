#!/bin/bash
# =============================================================================
# COMPASS HPC Step 6: Serve Open Weights with vLLM
# =============================================================================
#
# PURPOSE:
#   Start an OpenAI-compatible vLLM server for a HuggingFace model and print the
#   exact base URL the COMPASS dashboard (or main.py) should point at.
#
#   One script, three hardware shapes:
#     - one GPU                    TP_SIZE=1, PP_SIZE=1
#     - several GPUs on one node   TP_SIZE=<gpus per node>   (tensor parallel)
#     - several nodes              PP_SIZE=<nodes>           (pipeline parallel + Ray)
#
#   Three runtimes: the Apptainer image (cluster default), the venv from
#   02_setup_environment.sh (native), or the Docker GPU image (workstation).
#
# USAGE:
#   cd ~/compass_pipeline/multi_agent_decision_support_system
#
#   # 1. Single GPU, model from step 03 or straight from HuggingFace
#   bash src/full_stack/backend/hpc/06_serve_vllm.sh
#
#   # 2. Four GPUs on one node (tensor parallel)
#   COMPASS_SLURM_GRES=gpu:l40s:4 TP_SIZE=4 \
#     bash src/full_stack/backend/hpc/06_serve_vllm.sh
#
#   # 3. Two nodes with four GPUs each (tensor parallel inside a node,
#   #    pipeline parallel across nodes, wired together by Ray)
#   COMPASS_SLURM_NODES=2 COMPASS_SLURM_GRES=gpu:l40s:4 TP_SIZE=4 PP_SIZE=2 \
#     bash src/full_stack/backend/hpc/06_serve_vllm.sh
#
#   # 4. A different model
#   MODEL_ID=Qwen/Qwen3-32B-AWQ bash src/full_stack/backend/hpc/06_serve_vllm.sh
#
#   # 5. On a workstation with Docker and no scheduler
#   RUNTIME=docker bash src/full_stack/backend/hpc/06_serve_vllm.sh
#
# OPTIONAL ENV:
#   MODEL_ID              HuggingFace repo id or local path (default Qwen/Qwen3-14B-AWQ)
#   SERVED_MODEL_NAME     name clients must ask for (default: MODEL_ID)
#   SERVE_PORT            server port (default 8000)
#   SERVE_HOST            bind address (default 0.0.0.0)
#   SERVE_API_KEY         require this key from clients (default local-vllm)
#   TP_SIZE               tensor parallel size (default auto = GPUs on the node)
#   PP_SIZE               pipeline parallel size (default auto = number of nodes)
#   MAX_MODEL_LEN         context window served (default 32768)
#   GPU_MEM_UTIL          fraction of VRAM vLLM may use (default 0.90)
#   EXTRA_ARGS            free-form extra flags passed to `vllm serve`
#   TRUST_REMOTE_CODE     1 allows code shipped in the model repo, 0 refuses (default 1)
#   RUNTIME               auto|apptainer|native|docker (default auto)
#   COMPASS_SERVE_VENV    auto|none|force: whether to source VENV_DIR (default auto)
#   DOCKER_IMAGE_TAG      image used by RUNTIME=docker (default compass-engine:gpu)
#   STARTUP_TIMEOUT       seconds to wait for /health before giving up (default 1800)
#   COMPASS_SLURM_*       Slurm resource request, see below
#
# =============================================================================

#SBATCH --job-name=compass_vllm
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm_serve_%j.out
#SBATCH --error=logs/vllm_serve_%j.err

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR="${PROJECT_DIR:-${HOME}/compass_pipeline/multi_agent_decision_support_system}"
LOG_DIR="${PROJECT_DIR}/logs"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/compass_containers/pytorch_24.01.sif}"
VENV_DIR="${VENV_DIR:-${HOME}/compass_venv}"
MODELS_DIR="${MODELS_DIR:-${HOME}/compass_models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${MODELS_DIR}/hf_cache}"

# Model and server (override via env)
: "${MODEL_ID:=Qwen/Qwen3-14B-AWQ}"
: "${SERVED_MODEL_NAME:=${MODEL_ID}}"
: "${SERVE_PORT:=8000}"
: "${SERVE_HOST:=0.0.0.0}"
: "${SERVE_API_KEY:=local-vllm}"
: "${MAX_MODEL_LEN:=32768}"
: "${GPU_MEM_UTIL:=0.90}"
: "${EXTRA_ARGS:=}"
: "${TRUST_REMOTE_CODE:=1}"
: "${STARTUP_TIMEOUT:=1800}"

# Parallelism. "auto" resolves on the compute node from the actual allocation.
: "${TP_SIZE:=auto}"
: "${PP_SIZE:=auto}"

# Runtime
: "${RUNTIME:=auto}"
# Whether the server command is prefixed with `source ${VENV_DIR}/bin/activate`.
# See the resolution below: "auto" only uses it when the runtime has no vLLM of
# its own, "none" never does, "force" always does.
: "${COMPASS_SERVE_VENV:=auto}"
: "${DOCKER_IMAGE_TAG:=compass-engine:gpu}"
: "${RAY_PORT:=6379}"

# Slurm resource request used by the login-node auto-submit below. The #SBATCH
# lines above are the defaults for a direct `sbatch`; these variables let a
# multi-GPU or multi-node serve run without editing this file.
: "${COMPASS_SLURM_PARTITION:=main}"
: "${COMPASS_SLURM_NODES:=1}"
: "${COMPASS_SLURM_NTASKS_PER_NODE:=1}"
: "${COMPASS_SLURM_CPUS_PER_TASK:=16}"
: "${COMPASS_SLURM_MEM:=64G}"
: "${COMPASS_SLURM_GRES:=gpu:l40s:1}"
: "${COMPASS_SLURM_TIME:=08:00:00}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,57p' "$0"
    exit 0
fi

# ─── Header / Fingerprint ───────────────────────────────────────────────────
echo "============================================="
echo " COMPASS HPC: vLLM Server"
echo "============================================="
echo ""
echo "SCRIPT_PATH:  $0"
echo "Hostname:     $(hostname)"
echo "User:         $(whoami)"
echo "Date:         $(date)"
echo ""

# ─── Auto-Submit to Compute Node ───────────────────────────────────────────
CURRENT_HOST="$(hostname)"
if [[ "${CURRENT_HOST}" == login* ]] && [[ -z "${SLURM_JOB_ID:-}" ]] && [[ "${RUNTIME}" != "docker" ]]; then
    mkdir -p "${LOG_DIR}"
    echo "⚠  Login node detected. GPUs and Apptainer live on compute nodes."
    echo "   Auto-submitting this script as a Slurm job..."
    echo ""

    JOB_ID="$(sbatch --parsable \
        --job-name="compass_vllm" \
        --output="${LOG_DIR}/vllm_serve_%j.out" \
        --error="${LOG_DIR}/vllm_serve_%j.err" \
        --partition="${COMPASS_SLURM_PARTITION}" \
        --nodes="${COMPASS_SLURM_NODES}" \
        --ntasks-per-node="${COMPASS_SLURM_NTASKS_PER_NODE}" \
        --cpus-per-task="${COMPASS_SLURM_CPUS_PER_TASK}" \
        --mem="${COMPASS_SLURM_MEM}" \
        --gres="${COMPASS_SLURM_GRES}" \
        --time="${COMPASS_SLURM_TIME}" \
        --chdir="${PROJECT_DIR}" \
        --export=ALL \
        "$0")"

    echo "✓ Server job submitted! Job ID: ${JOB_ID}"
    echo ""
    echo "  The base URL is printed in the log as soon as the server answers:"
    echo "    tail -f ${LOG_DIR}/vllm_serve_${JOB_ID}.out"
    echo "    cat ${LOG_DIR}/vllm_endpoint_${JOB_ID}.txt"
    echo ""
    echo "  Queue:"
    echo "    squeue -u $(whoami)"
    echo ""
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# FROM HERE: Running where the GPUs are (compute node, or a local workstation)
# ═════════════════════════════════════════════════════════════════════════════

mkdir -p "${LOG_DIR}" "${HF_CACHE_DIR}"

# ─── Resolve the model location ────────────────────────────────────────────
# Step 03 stores weights as <MODELS_DIR>/<owner>_<repo>. Preferring that copy
# keeps the server working on compute nodes without outbound internet.
LOCAL_MODEL_DIR="${MODELS_DIR}/${MODEL_ID//\//_}"
if [[ -d "${MODEL_ID}" ]]; then
    # Absolute, because the docker branch below binds this path into the container.
    MODEL_PATH="$(cd "${MODEL_ID}" && pwd)"
    MODEL_SOURCE="local path"
elif [[ -d "${LOCAL_MODEL_DIR}" ]]; then
    MODEL_PATH="${LOCAL_MODEL_DIR}"
    MODEL_SOURCE="local copy from step 03"
else
    MODEL_PATH="${MODEL_ID}"
    MODEL_SOURCE="HuggingFace download into ${HF_CACHE_DIR}"
fi

# ─── Resolve the allocation ────────────────────────────────────────────────
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"

count_gpus() {
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        echo "${SLURM_GPUS_ON_NODE}"
        return
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}'
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local detected
        detected="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
        echo "${detected:-1}"
        return
    fi
    echo 1
}

GPUS_PER_NODE="$(count_gpus)"
if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || (( GPUS_PER_NODE < 1 )); then
    GPUS_PER_NODE=1
fi

if [[ "${TP_SIZE}" == "auto" ]]; then
    TP_SIZE="${GPUS_PER_NODE}"
fi
if [[ "${PP_SIZE}" == "auto" ]]; then
    PP_SIZE="${NUM_NODES}"
fi

if (( TP_SIZE > GPUS_PER_NODE )); then
    echo "✗ ERROR: TP_SIZE=${TP_SIZE} exceeds the ${GPUS_PER_NODE} GPU(s) visible on this node."
    echo "  Request more GPUs (COMPASS_SLURM_GRES=gpu:<type>:${TP_SIZE}) or lower TP_SIZE."
    exit 1
fi

NODE_ADDR="$(hostname -f 2>/dev/null || hostname)"

echo "─── Serve plan ────────────────────────────────────────────"
echo "  Model:             ${MODEL_ID}"
echo "  Model source:      ${MODEL_SOURCE}"
echo "  Served as:         ${SERVED_MODEL_NAME}"
echo "  Nodes:             ${NUM_NODES}"
echo "  GPUs per node:     ${GPUS_PER_NODE}"
echo "  Tensor parallel:   ${TP_SIZE}"
echo "  Pipeline parallel: ${PP_SIZE}"
echo "  Context window:    ${MAX_MODEL_LEN}"
echo "  GPU memory util:   ${GPU_MEM_UTIL}"
echo "  Bind:              ${SERVE_HOST}:${SERVE_PORT}"
echo "  Node address:      ${NODE_ADDR}"
echo ""

# ─── Resolve the runtime ───────────────────────────────────────────────────
RESOLVED_RUNTIME="${RUNTIME}"
if [[ "${RESOLVED_RUNTIME}" == "auto" ]]; then
    if command -v apptainer >/dev/null 2>&1 && [[ -f "${CONTAINER_IMAGE}" ]]; then
        RESOLVED_RUNTIME="apptainer"
    elif [[ -x "${VENV_DIR}/bin/python3" ]]; then
        RESOLVED_RUNTIME="native"
    elif command -v docker >/dev/null 2>&1; then
        RESOLVED_RUNTIME="docker"
    else
        echo "✗ ERROR: no usable runtime found."
        echo "  Expected one of:"
        echo "    apptainer + ${CONTAINER_IMAGE}   (bash src/full_stack/backend/hpc/02_setup_environment.sh)"
        echo "    venv at ${VENV_DIR}"
        echo "    docker with ${DOCKER_IMAGE_TAG}"
        exit 1
    fi
fi
echo "  Runtime:           ${RESOLVED_RUNTIME}"
echo ""

RUNTIME_EXEC=()
case "${RESOLVED_RUNTIME}" in
    native)
        RUNTIME_EXEC=(bash -lc)
        ;;
    apptainer)
        if ! command -v apptainer >/dev/null 2>&1; then
            echo "✗ ERROR: RUNTIME=apptainer but apptainer is not in PATH on $(hostname)."
            exit 1
        fi
        if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
            echo "✗ ERROR: container image not found: ${CONTAINER_IMAGE}"
            echo "  Build it: apptainer build ${CONTAINER_IMAGE} src/full_stack/backend/hpc/apptainer.def"
            exit 1
        fi
        # --nv exposes the host NVIDIA driver inside the container. Without it
        # torch sees no GPU at all.
        RUNTIME_EXEC=(
            apptainer exec --nv
            --bind "${PROJECT_DIR}:${PROJECT_DIR}"
            --bind "${MODELS_DIR}:${MODELS_DIR}"
            --bind "${HOME}:${HOME}"
            --env "HF_HOME=${HF_CACHE_DIR}"
            --env "HF_HUB_CACHE=${HF_CACHE_DIR}/hub"
            --env "PYTHONUNBUFFERED=1"
            --env "HF_TOKEN=${HF_TOKEN:-}"
            "${CONTAINER_IMAGE}"
            bash -lc
        )
        ;;
    docker)
        if ! command -v docker >/dev/null 2>&1; then
            echo "✗ ERROR: RUNTIME=docker but docker is not in PATH."
            exit 1
        fi
        if (( NUM_NODES > 1 )); then
            echo "✗ ERROR: RUNTIME=docker covers one machine only."
            echo "  For several nodes use RUNTIME=apptainer or RUNTIME=native under Slurm."
            exit 1
        fi
        ;;
    *)
        echo "✗ ERROR: unknown RUNTIME='${RUNTIME}' (expected auto|apptainer|native|docker)."
        exit 1
        ;;
esac

# ─── Resolve the Python environment ────────────────────────────────────────
# The venv from 02_setup_environment.sh was created inside the NGC PyTorch image
# with --system-site-packages, so sourcing it inside an image that ships its own
# /opt/venv (apptainer.def, docker/Dockerfile.gpu) puts the wrong interpreter
# first on PATH and hides that image's vLLM. The prefix is therefore tied to the
# image in use, not merely to the venv existing on the host.
VENV_PREFIX=""
VENV_REASON="none (the runtime carries its own environment)"

venv_activate_prefix() {
    if [[ ! -x "${VENV_DIR}/bin/python3" ]]; then
        echo "✗ ERROR: COMPASS_SERVE_VENV=${COMPASS_SERVE_VENV} but no venv at ${VENV_DIR}."
        echo "  Create it with: bash src/full_stack/backend/hpc/02_setup_environment.sh"
        exit 1
    fi
    VENV_PREFIX="source '${VENV_DIR}/bin/activate'; "
    VENV_REASON="${VENV_DIR}"
}

case "${COMPASS_SERVE_VENV}" in
    none)
        VENV_REASON="none (COMPASS_SERVE_VENV=none)"
        ;;
    force)
        venv_activate_prefix
        VENV_REASON="${VENV_DIR} (COMPASS_SERVE_VENV=force)"
        ;;
    auto)
        case "${RESOLVED_RUNTIME}" in
            native)
                if [[ -x "${VENV_DIR}/bin/python3" ]]; then
                    venv_activate_prefix
                else
                    VENV_REASON="none (no venv at ${VENV_DIR}, using the system Python)"
                fi
                ;;
            apptainer)
                # compass_gpu.sif answers this; the NGC PyTorch image does not,
                # and that one does need the step-02 venv.
                if apptainer exec "${CONTAINER_IMAGE}" test -x /opt/venv/bin/vllm >/dev/null 2>&1; then
                    VENV_REASON="/opt/venv inside ${CONTAINER_IMAGE##*/}"
                elif [[ -x "${VENV_DIR}/bin/python3" ]]; then
                    venv_activate_prefix
                else
                    VENV_REASON="none (no /opt/venv in the image and no venv at ${VENV_DIR})"
                fi
                ;;
            docker)
                VENV_REASON="/opt/venv inside ${DOCKER_IMAGE_TAG}"
                ;;
        esac
        ;;
    *)
        echo "✗ ERROR: unknown COMPASS_SERVE_VENV='${COMPASS_SERVE_VENV}' (expected auto|none|force)."
        exit 1
        ;;
esac
echo "  Python env:        ${VENV_REASON}"
echo ""

# ─── Ray cluster for multi-node serving ────────────────────────────────────
# vLLM shards a model across nodes through Ray: one head, one worker per extra
# node, then the server is started on the head.
RAY_STARTED=0
HEAD_IP=""

start_ray_cluster() {
    if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
        echo "✗ ERROR: multi-node serving needs a Slurm allocation (SLURM_JOB_NODELIST is unset)."
        exit 1
    fi

    mapfile -t NODE_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    local head_node="${NODE_LIST[0]}"
    HEAD_IP="$(getent hosts "${head_node}" | awk '{print $1}' | head -1 || true)"
    if [[ -z "${HEAD_IP}" ]]; then
        HEAD_IP="${head_node}"
    fi

    echo "─── Ray cluster ───────────────────────────────────────────"
    echo "  Head:    ${head_node} (${HEAD_IP}:${RAY_PORT})"
    echo "  Workers: ${NODE_LIST[*]:1}"
    echo ""

    "${RUNTIME_EXEC[@]}" "${VENV_PREFIX}ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats"
    RAY_STARTED=1

    local node
    for node in "${NODE_LIST[@]:1}"; do
        echo "  Starting Ray worker on ${node}..."
        srun --nodes=1 --ntasks=1 -w "${node}" \
            "${RUNTIME_EXEC[@]}" "${VENV_PREFIX}ray start --address='${HEAD_IP}:${RAY_PORT}' --num-gpus=${GPUS_PER_NODE} --disable-usage-stats --block" &
        sleep 5
    done

    echo "  Waiting for ${NUM_NODES} nodes to join..."
    sleep 20
    "${RUNTIME_EXEC[@]}" "${VENV_PREFIX}ray status" || true
    echo ""
}

stop_ray_cluster() {
    if (( RAY_STARTED == 1 )); then
        "${RUNTIME_EXEC[@]}" "${VENV_PREFIX}ray stop" >/dev/null 2>&1 || true
        RAY_STARTED=0
    fi
}

SERVER_PID=""
DOCKER_NAME="compass-vllm-$$"

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ "${RESOLVED_RUNTIME}" == "docker" ]]; then
        docker rm -f "${DOCKER_NAME}" >/dev/null 2>&1 || true
    fi
    stop_ray_cluster
}
trap cleanup EXIT INT TERM

if (( NUM_NODES > 1 )); then
    start_ray_cluster
fi

# ─── Build the server command ──────────────────────────────────────────────
SERVE_CMD="vllm serve '${MODEL_PATH}'"
SERVE_CMD+=" --host '${SERVE_HOST}' --port ${SERVE_PORT}"
SERVE_CMD+=" --served-model-name '${SERVED_MODEL_NAME}'"
SERVE_CMD+=" --tensor-parallel-size ${TP_SIZE}"
SERVE_CMD+=" --pipeline-parallel-size ${PP_SIZE}"
SERVE_CMD+=" --max-model-len ${MAX_MODEL_LEN}"
SERVE_CMD+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
# Same default and same off switch as docker/entrypoint.gpu.sh and the
# %runscript in apptainer.def, so the three paths serve identical flags.
case "${TRUST_REMOTE_CODE}" in
    0|false|no) ;;
    *) SERVE_CMD+=" --trust-remote-code" ;;
esac
if [[ -n "${SERVE_API_KEY}" ]]; then
    SERVE_CMD+=" --api-key '${SERVE_API_KEY}'"
fi
if (( NUM_NODES > 1 )); then
    SERVE_CMD+=" --distributed-executor-backend ray"
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
    SERVE_CMD+=" ${EXTRA_ARGS}"
fi

echo "─── Starting server ───────────────────────────────────────"
if [[ "${RESOLVED_RUNTIME}" == "docker" ]]; then
    echo "  ${DOCKER_IMAGE_TAG} serve (the image builds the same flags from COMPASS_VLLM_*)"
else
    echo "  ${SERVE_CMD}"
fi
echo ""

START_TIME=${SECONDS}

if [[ "${RESOLVED_RUNTIME}" == "docker" ]]; then
    GPU_FLAG="--gpus all"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        # The inner quotes are part of the value Docker expects; without them it
        # splits the device list on the comma and rejects the request.
        GPU_FLAG="--gpus \"device=${CUDA_VISIBLE_DEVICES}\""
    fi
    # MODELS_DIR is bound under its own path so the resolved local copy is
    # readable inside the container; without it the server would ignore the
    # weights step 03 downloaded and try to fetch them again.
    MODEL_MOUNTS=(-v "${MODELS_DIR}:${MODELS_DIR}")
    if [[ -d "${MODEL_PATH}" && "${MODEL_PATH}" != "${MODELS_DIR}"* ]]; then
        MODEL_MOUNTS+=(-v "${MODEL_PATH}:${MODEL_PATH}")
    fi
    # shellcheck disable=SC2086
    docker run --rm --name "${DOCKER_NAME}" \
        ${GPU_FLAG} \
        --shm-size=16g \
        -p "${SERVE_PORT}:${SERVE_PORT}" \
        -v "${HF_CACHE_DIR}:/home/compass/.cache/huggingface" \
        "${MODEL_MOUNTS[@]}" \
        -e COMPASS_VLLM_MODEL="${MODEL_PATH}" \
        -e COMPASS_VLLM_HOST="${SERVE_HOST}" \
        -e COMPASS_VLLM_PORT="${SERVE_PORT}" \
        -e COMPASS_VLLM_TP="${TP_SIZE}" \
        -e COMPASS_VLLM_PP="${PP_SIZE}" \
        -e COMPASS_VLLM_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        -e COMPASS_VLLM_GPU_MEM_UTIL="${GPU_MEM_UTIL}" \
        -e COMPASS_VLLM_SERVED_NAME="${SERVED_MODEL_NAME}" \
        -e COMPASS_VLLM_API_KEY="${SERVE_API_KEY}" \
        -e COMPASS_VLLM_EXTRA_ARGS="${EXTRA_ARGS}" \
        -e COMPASS_VLLM_TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        "${DOCKER_IMAGE_TAG}" serve &
    SERVER_PID=$!
else
    "${RUNTIME_EXEC[@]}" "${VENV_PREFIX}${SERVE_CMD}" &
    SERVER_PID=$!
fi

# ─── Wait for the server to answer ─────────────────────────────────────────
probe_health() {
    local url="http://127.0.0.1:${SERVE_PORT}/health"
    if command -v curl >/dev/null 2>&1; then
        curl -fsS -o /dev/null --max-time 5 "${url}"
        return $?
    fi
    python3 - "${url}" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=5)
except Exception:
    sys.exit(1)
PY
}

echo "  Waiting for the server to load weights (timeout ${STARTUP_TIMEOUT}s)..."
READY=0
DEADLINE=$((SECONDS + STARTUP_TIMEOUT))
while (( SECONDS < DEADLINE )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "✗ ERROR: the server process exited before it became healthy."
        echo "  Check the log above for the vLLM traceback."
        wait "${SERVER_PID}" || true
        exit 1
    fi
    if probe_health; then
        READY=1
        break
    fi
    sleep 10
done

if (( READY == 0 )); then
    echo "✗ ERROR: server did not answer /health within ${STARTUP_TIMEOUT}s."
    echo "  Raise STARTUP_TIMEOUT for very large models, or lower MAX_MODEL_LEN."
    exit 1
fi

ELAPSED=$((SECONDS - START_TIME))
BASE_URL="http://${NODE_ADDR}:${SERVE_PORT}/v1"
ENDPOINT_FILE="${LOG_DIR}/vllm_endpoint_${SLURM_JOB_ID:-local}.txt"

{
    echo "base_url=${BASE_URL}"
    echo "model=${SERVED_MODEL_NAME}"
    echo "api_key=${SERVE_API_KEY}"
    echo "node=${NODE_ADDR}"
    echo "port=${SERVE_PORT}"
    echo "job_id=${SLURM_JOB_ID:-local}"
} > "${ENDPOINT_FILE}"
# The file carries the server key, so keep it to the owner.
chmod 600 "${ENDPOINT_FILE}" 2>/dev/null || true

echo ""
echo "============================================="
echo " ✓ vLLM SERVER READY (${ELAPSED}s)"
echo "============================================="
echo ""
echo " Base URL for the dashboard:"
echo "   ${BASE_URL}"
echo ""
echo " Model name clients must request:"
echo "   ${SERVED_MODEL_NAME}"
echo ""
echo " Point COMPASS at it (dashboard: Settings, Connection, Base URL):"
echo "   export OPENROUTER_BASE_URL='${BASE_URL}'"
echo "   export OPENROUTER_API_KEY='${SERVE_API_KEY}'"
echo "   python3 main.py <participant_dir> --backend openrouter --model '${SERVED_MODEL_NAME}' ..."
echo ""
echo " From a laptop, tunnel through the login node first:"
echo "   ssh -N -L ${SERVE_PORT}:${NODE_ADDR}:${SERVE_PORT} ${USER}@<cluster-login-host>"
echo "   then use http://localhost:${SERVE_PORT}/v1"
echo ""
echo " Endpoint written to:"
echo "   ${ENDPOINT_FILE}"
echo ""
echo " Stop the server with: scancel ${SLURM_JOB_ID:-<job id>}"
echo "============================================="
echo ""

wait "${SERVER_PID}"
