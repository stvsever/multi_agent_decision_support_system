#!/bin/bash
# =============================================================================
# COMPASS HPC Deployment Helper
# =============================================================================
# This script automates transferring your code to a configured HPC cluster
# and initiating the setup process.
#
# USAGE:
#   bash src/full_stack/backend/hpc/00_deploy_and_run.sh
# =============================================================================

# Configuration (override via environment)
: "${HPC_USER:=...}"
: "${HPC_HOST:=...}"
HPC_DIR="~/compass_pipeline"
# The folder name every other script expects under HPC_DIR (their PROJECT_DIR),
# whatever the local clone is called.
: "${HPC_PROJECT_NAME:=multi_agent_decision_support_system}"
# 1 (default) leaves out the maintainer's local-only folders (validation/,
# marketing/, report/: gitignored, about 1.5 GB); 0 copies them too.
: "${DEPLOY_EXCLUDE_LOCAL_ONLY:=1}"

# Local paths (override via environment if needed)
# - Defaults are derived relative to this repo.
# - Examples:
#     LOCAL_CODE_DIR=/path/to/multi_agent_decision_support_system LOCAL_DATA_DIR=/path/to/HPC_data bash src/full_stack/backend/hpc/00_deploy_and_run.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
: "${LOCAL_CODE_DIR:=${REPO_ROOT}}"
: "${LOCAL_DATA_DIR:=${REPO_ROOT}/../data/__FEATURES__/HPC_data}"

echo "========================================================"
echo " 🚀 COMPASS Deployment to HPC"
echo "========================================================"
echo "Target: ${HPC_USER}@${HPC_HOST}"
echo ""
echo "Note: You will be prompted for your password multiple times"
echo "      (once for each command) unless you have SSH keys set up."
echo "      Password: ..."
echo "========================================================"
echo ""

# 1. Create remote directory structure
echo "─── [1/3] Creating remote directories..."
ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p ${HPC_DIR}/data/__FEATURES__ 2>/dev/null" || true

# 2. Upload Code
echo ""
echo "─── [2/3] Uploading Project Code..."
# We exclude __pycache__, .git, and local logs to save time/space
# Using rsync if available, else scp
# API keys (.env) never travel with the bulk copy: rsync -a would keep a local
# mode such as 644, which on a shared cluster lets other users read them. They
# are offered separately below, written with mode 600.
LOCAL_ONLY_EXCLUDES=()
if [[ "${DEPLOY_EXCLUDE_LOCAL_ONLY}" == "1" ]]; then
    LOCAL_ONLY_EXCLUDES=(--exclude '/validation' --exclude '/marketing' --exclude '/report')
fi
if command -v rsync &> /dev/null; then
    rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude '.DS_Store' \
        --exclude '.env' --exclude 'node_modules' --exclude '.venv' \
        ${LOCAL_ONLY_EXCLUDES[@]+"${LOCAL_ONLY_EXCLUDES[@]}"} \
        "${LOCAL_CODE_DIR}/" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${HPC_PROJECT_NAME}/"
else
    echo "⚠  rsync not found: scp copies everything, including any .env; check its mode on the cluster."
    scp -r "${LOCAL_CODE_DIR}" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${HPC_PROJECT_NAME}"
fi

if [[ -f "${LOCAL_CODE_DIR}/.env" ]] && command -v rsync &> /dev/null; then
    read -p "Upload .env (API keys, needed for a TypeSafe Jev Predictor) with mode 600? [y/N]: " upload_env
    if [[ "$upload_env" =~ ^[Yy]$ ]]; then
        rsync -az --chmod=F600 "${LOCAL_CODE_DIR}/.env" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/${HPC_PROJECT_NAME}/.env"
        echo "✓ .env uploaded (mode 600)"
    fi
fi

# 3. Upload Data (Optional prompt)
echo ""
echo "─── [3/3] Participant Data..."
if [ -d "${LOCAL_DATA_DIR}" ]; then
    echo "Found local data at: ${LOCAL_DATA_DIR}"
    read -p "Upload participant data now? (This might take a while) [y/N]: " upload_data
    if [[ "$upload_data" =~ ^[Yy]$ ]]; then
        echo "Uploading data..."
        if command -v rsync &> /dev/null; then
             rsync -avz "${LOCAL_DATA_DIR}" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/__FEATURES__/"
        else
             scp -r "${LOCAL_DATA_DIR}" "${HPC_USER}@${HPC_HOST}:${HPC_DIR}/data/__FEATURES__/"
        fi
    else
        echo "Skipping data upload."
    fi
else
    echo "⚠️  Local data directory not found at: ${LOCAL_DATA_DIR}"
    echo "   Please make sure to upload data manually."
fi

echo ""
echo "========================================================"
echo "Deployment Complete!"
echo "========================================================"
echo "To start the pipeline, run these commands inside the HPC:"
echo ""
echo "  ssh ${HPC_USER}@${HPC_HOST}"
echo "  cd ${HPC_DIR}/${HPC_PROJECT_NAME}"
echo "  bash src/full_stack/backend/hpc/01_check_status.sh"
echo ""
echo "Connecting you now..."
echo "========================================================"
ssh "${HPC_USER}@${HPC_HOST}"
