#!/usr/bin/env bash
# Role dispatcher for the GPU image: the same image runs the vLLM server and the
# dashboard, so the first argument picks which one starts.
#
#   dashboard   COMPASS dashboard on COMPASS_UI_PORT (default)
#   serve       vLLM OpenAI-compatible server on COMPASS_VLLM_PORT
#   --*         forwarded to main.py --ui
#   anything    executed verbatim, so `bash` and `python3 main.py ...` still work
set -euo pipefail

export COMPASS_UI_HOST="${COMPASS_UI_HOST:-0.0.0.0}"
export COMPASS_UI_PORT="${COMPASS_UI_PORT:-5005}"

#: Where the HEALTHCHECK looks for the role's probe URL. A file, because the
#: probe is a sibling process the daemon builds from the container's configured
#: environment: nothing this script exports can reach it.
HEALTH_URL_FILE="${COMPASS_HEALTH_URL_FILE:-/tmp/compass-health-url}"

announce_health_url() {
    # A read-only or full filesystem is not worth failing a start over: the
    # probe falls back to the dashboard default when the file is absent.
    printf '%s' "$1" > "${HEALTH_URL_FILE}" 2>/dev/null || true
}

start_dashboard() {
    announce_health_url "http://127.0.0.1:${COMPASS_UI_PORT}/api/health"
    exec python3 main.py --ui "$@"
}

serve_vllm() {
    local model="${COMPASS_VLLM_MODEL:-}"
    if [ -z "${model}" ]; then
        echo "✗ ERROR: COMPASS_VLLM_MODEL is not set." >&2
        echo "  Example: -e COMPASS_VLLM_MODEL=Qwen/Qwen3-14B-AWQ" >&2
        exit 2
    fi

    # The dashboard port the probe defaults to is dead in this role, so the
    # probe has to be told where to look. It cannot be told with an export:
    # the daemon builds the probe as a sibling process from the container's
    # configured environment, not as a child of this shell, so a runtime export
    # never reaches it. The filesystem is shared, so the URL is written there
    # and the HEALTHCHECK reads the file.
    export COMPASS_HEALTHCHECK_URL="${COMPASS_HEALTHCHECK_URL:-http://127.0.0.1:${COMPASS_VLLM_PORT:-8000}/health}"
    announce_health_url "${COMPASS_HEALTHCHECK_URL}"

    # Keep this list byte-for-byte in step with the %runscript in
    # src/full_stack/backend/hpc/apptainer.def: the two roles are documented as
    # twins, so a model that serves on the cluster must serve here too.
    local args=(
        "${model}"
        --host "${COMPASS_VLLM_HOST:-0.0.0.0}"
        --port "${COMPASS_VLLM_PORT:-8000}"
        --tensor-parallel-size "${COMPASS_VLLM_TP:-1}"
        --pipeline-parallel-size "${COMPASS_VLLM_PP:-1}"
        --gpu-memory-utilization "${COMPASS_VLLM_GPU_MEM_UTIL:-0.90}"
        --max-model-len "${COMPASS_VLLM_MAX_MODEL_LEN:-32768}"
    )
    # Repositories that ship their own modeling code need this. Set the variable
    # to 0 to refuse to execute code from the model repository.
    case "${COMPASS_VLLM_TRUST_REMOTE_CODE:-1}" in
        0|false|no) ;;
        *) args+=(--trust-remote-code) ;;
    esac
    if [ -n "${COMPASS_VLLM_SERVED_NAME:-}" ]; then
        args+=(--served-model-name "${COMPASS_VLLM_SERVED_NAME}")
    fi
    if [ -n "${COMPASS_VLLM_API_KEY:-}" ]; then
        args+=(--api-key "${COMPASS_VLLM_API_KEY}")
    fi
    if [ -n "${COMPASS_VLLM_QUANT:-}" ]; then
        args+=(--quantization "${COMPASS_VLLM_QUANT}")
    fi

    echo "Serving ${model} with vLLM"
    echo "  tensor parallel:   ${COMPASS_VLLM_TP:-1}"
    echo "  pipeline parallel: ${COMPASS_VLLM_PP:-1}"
    echo "  health probe:      ${COMPASS_HEALTHCHECK_URL}"
    echo "  base URL inside the compose network: http://vllm:${COMPASS_VLLM_PORT:-8000}/v1"
    echo "  base URL from the host:              http://localhost:${COMPASS_VLLM_PORT:-8000}/v1"

    # Word splitting on COMPASS_VLLM_EXTRA_ARGS is intended: it is a free-form
    # passthrough for vLLM flags this wrapper does not model. Anything after
    # `serve` on the command line is appended last so it wins.
    # shellcheck disable=SC2086
    exec vllm serve "${args[@]}" ${COMPASS_VLLM_EXTRA_ARGS:-} "$@"
}

if [ "$#" -eq 0 ]; then
    start_dashboard
fi

case "$1" in
    dashboard)
        shift
        start_dashboard "$@"
        ;;
    serve)
        shift
        serve_vllm "$@"
        ;;
    --*)
        start_dashboard "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
