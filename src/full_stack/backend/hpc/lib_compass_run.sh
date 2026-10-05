#!/bin/bash
# =============================================================================
# COMPASS HPC: shared run options for 04_submit_single.sh and 05_submit_batch.sh
# =============================================================================
# Sourced, not executed. Defines the Predictor, routing and decision-model
# options both submission scripts pass to main.py, and the API key check.
#
# OPTIONAL ENV (all map to main.py flags; see `python3 main.py --help`):
#   PREDICTOR_MODEL          Predictor only, for example typesafe/jev-1.13 (a
#                            structured decision model called over the network;
#                            the local --model stays the companion for every
#                            other role). Empty: the local model predicts too.
#   ORCHESTRATION            auto (default) | always | never
#   ORCHESTRATION_THRESHOLD  tokens above which auto orchestrates (0 = the
#                            Predictor input budget)
#   DECISION_PROVIDER        openrouter (OPENROUTER_API_KEY, default) | typesafe
#                            (TYPESAFE_API_KEY)
#   DECISION_CHOICE_ORDERS   option orders per multiclass Choice (default 3)
#   DECISION_SCORE_LEVELS    levels per regression Score, 2 to 10 (default 10)
#   DECISION_REFINE          1 (default) or 0: zoomed second Score pass
#   PREFLIGHT_CHECK          1 (default): main.py --check_config before the run
#                            (offline; refuses a run that would fail to start)
#   COMPASS_ENV_FILE         KEY=value file with the API keys (default
#                            ${PROJECT_DIR}/.env, which the engine also reads by
#                            itself). Keep it chmod 600. Keys are never echoed,
#                            never put on a command line and never passed with
#                            `--env KEY=value`, so they stay out of `ps`, the
#                            Slurm job record and the job logs.
# =============================================================================

: "${PREDICTOR_MODEL:=}"
: "${ORCHESTRATION:=auto}"
: "${ORCHESTRATION_THRESHOLD:=0}"
: "${DECISION_PROVIDER:=openrouter}"
: "${DECISION_CHOICE_ORDERS:=}"
: "${DECISION_SCORE_LEVELS:=}"
: "${DECISION_REFINE:=1}"
: "${PREFLIGHT_CHECK:=1}"
: "${COMPASS_ENV_FILE:=${PROJECT_DIR}/.env}"

# True when the id names a structured decision model (only the Predictor may use one).
# Same rule as the engine (agents/decision/registry.py): typesafe/jev-<version> or
# the bare jev-<version>, but not the jev-router LLM.
compass_is_decision_model() {
    local id
    id="$(printf '%s' "${1}" | tr '[:upper:]' '[:lower:]')"
    id="${id#\~}"
    [[ "${id}" =~ ^(typesafe/)?jev-[a-z0-9.-]+$ && ! "${id}" =~ ^(typesafe/)?jev-router ]]
}

# The main.py flags for the options above, single-quoted for the inner shell.
compass_routing_flags() {
    local flags="--orchestration '${ORCHESTRATION}'"
    if [[ "${ORCHESTRATION_THRESHOLD}" =~ ^[0-9]+$ && "${ORCHESTRATION_THRESHOLD}" != "0" ]]; then
        flags+=" --orchestration_threshold ${ORCHESTRATION_THRESHOLD}"
    fi
    if [[ -n "${PREDICTOR_MODEL}" ]]; then
        flags+=" --predictor_model '${PREDICTOR_MODEL}'"
        if compass_is_decision_model "${PREDICTOR_MODEL}"; then
            flags+=" --decision_provider '${DECISION_PROVIDER}'"
            [[ -n "${DECISION_CHOICE_ORDERS}" ]] && flags+=" --decision_choice_orders ${DECISION_CHOICE_ORDERS}"
            [[ -n "${DECISION_SCORE_LEVELS}" ]] && flags+=" --decision_score_levels ${DECISION_SCORE_LEVELS}"
            if [[ "${DECISION_REFINE}" == "0" ]]; then
                flags+=" --no-decision_refine"
            fi
        fi
    fi
    echo "${flags}"
}

# The API key a decision-model Predictor needs, or nothing for a local-only run.
compass_required_key() {
    if [[ -n "${PREDICTOR_MODEL}" ]] && compass_is_decision_model "${PREDICTOR_MODEL}"; then
        if [[ "${DECISION_PROVIDER}" == "typesafe" ]]; then
            echo "TYPESAFE_API_KEY"
        else
            echo "OPENROUTER_API_KEY"
        fi
    fi
}

# Fail early when the key a run needs is missing; print names only, never values.
compass_check_keys() {
    local key
    key="$(compass_required_key)"
    [[ -z "${key}" ]] && return 0
    if [[ -n "${!key:-}" ]]; then
        echo "✓ ${key} present in the environment"
        return 0
    fi
    if [[ -f "${COMPASS_ENV_FILE}" ]] && grep -qE "^[[:space:]]*(export[[:space:]]+)?${key}=." "${COMPASS_ENV_FILE}"; then
        local mode
        mode="$(stat -c '%a' "${COMPASS_ENV_FILE}" 2>/dev/null || stat -f '%Lp' "${COMPASS_ENV_FILE}" 2>/dev/null || echo '?')"
        if [[ "${mode}" != "600" && "${mode}" != "400" ]]; then
            echo "⚠  ${COMPASS_ENV_FILE} has mode ${mode}; run: chmod 600 '${COMPASS_ENV_FILE}'"
        fi
        echo "✓ ${key} present in ${COMPASS_ENV_FILE}"
        return 0
    fi
    echo "✗ ERROR: the Predictor ${PREDICTOR_MODEL} needs ${key}."
    echo "  Put '${key}=...' in ${COMPASS_ENV_FILE} (chmod 600) or export it before submitting."
    echo "  Compute nodes also need outbound HTTPS to the provider."
    return 1
}

# Shell line (for the inner `bash -lc`) that loads COMPASS_ENV_FILE when it is
# not the project .env the engine reads anyway. It never prints the values.
compass_env_loader() {
    if [[ -f "${COMPASS_ENV_FILE}" && "${COMPASS_ENV_FILE}" != "${PROJECT_DIR}/.env" ]]; then
        echo "set -a; . '${COMPASS_ENV_FILE}'; set +a"
    else
        echo ":"
    fi
}

compass_print_run_options() {
    echo "Predictor:          ${PREDICTOR_MODEL:-<the local model>}"
    echo "Orchestration:      ${ORCHESTRATION} (threshold ${ORCHESTRATION_THRESHOLD}; 0 = Predictor input budget)"
    if [[ -n "${PREDICTOR_MODEL}" ]] && compass_is_decision_model "${PREDICTOR_MODEL}"; then
        echo "Decision provider:  ${DECISION_PROVIDER} (key: $(compass_required_key))"
    fi
    echo "Preflight check:    ${PREFLIGHT_CHECK}"
}
