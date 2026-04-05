#!/bin/bash
#SBATCH --job-name=cluster_cmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=eng-research-gpu
#SBATCH --account=bbradlyn-ic
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --begin=now
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1

# =============================================================================
# Flexible cluster comparison runner
# =============================================================================
#
# This script runs the cluster_runner.py CLI with configurable commands/options.
#
# Usage via submit script:
#   CLUSTER_CMD="filling_cluster_sizes --L=48 --chi=64" ./scripts/submit_cluster_job.sh scripts/slurm/run_comparison.sbatch
#
# Or set in pyproject.toml:
#   [tool.cluster.run]
#   command = "python -m aah_code.cluster_model.cluster_runner"
#   args = ["filling_cluster_sizes", "--L=48"]
#
# Environment variables:
#   CLUSTER_CMD       - Full command args (e.g., "filling_cluster_sizes --L=48")
#   CLUSTER_FUNC      - Just the function name (e.g., "filling_cluster_sizes")
#   CLUSTER_L         - System size override
#   CLUSTER_CHI       - Bond dimension override
#   CLUSTER_FILLING   - Filling override (for fixed_supercluster)
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CLUSTER_RUN_CONFIG:-$SCRIPT_DIR/../../../../pyproject.toml}"

# Ensure environment modules are available
if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
fi

# Load modules
module load anaconda3/2024.10
module load cuda/12.6
module list

echo "Running on $(hostname)"
echo "Working directory: $SLURM_SUBMIT_DIR"
nvidia-smi || true

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# -----------------------------------------------------------------------------
# Python environment setup
# -----------------------------------------------------------------------------

pick_python() {
    local candidates=(python3.12 python3.11 python3 python)
    for bin in "${candidates[@]}"; do
        if command -v "$bin" >/dev/null 2>&1; then
            if "$bin" -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
                echo "$bin"
                return
            fi
        fi
    done
    echo ""
}

PYTHON_BIN="$(pick_python)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "No Python >=3.11 found" >&2
    exit 1
fi

echo "Using Python: $PYTHON_BIN"

VENV_DIR="$SLURM_SUBMIT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

pushd "$SLURM_SUBMIT_DIR/aah" >/dev/null
pip install -e .
popd >/dev/null

# Skip CuPy/CUDA extras — GPU eigensolver causes convergence failures
# and fallback overhead. All computation uses CPU (scipy/numpy).
# See used_data.md "RunPod execution notes" for details.

# Pin BLAS threads to avoid oversubscription on multi-core nodes
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Environment: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=$OMP_NUM_THREADS"

# -----------------------------------------------------------------------------
# Build the command
# -----------------------------------------------------------------------------

# Default command if nothing specified
DEFAULT_CMD="filling_cluster_sizes --L=24 --chi=32"

# Priority: CLUSTER_CONFIG (yaml) > CLUSTER_CMD > build from parts > default
if [[ -n "${CLUSTER_CONFIG:-}" ]]; then
    # Run from YAML config file
    CMD_ARGS="--config=${CLUSTER_CONFIG}"
elif [[ -n "${CLUSTER_CMD:-}" ]]; then
    CMD_ARGS="$CLUSTER_CMD"
else
    # Build from individual env vars
    CMD_ARGS="${CLUSTER_FUNC:-filling_cluster_sizes}"

    [[ -n "${CLUSTER_L:-}" ]] && CMD_ARGS+=" --L=${CLUSTER_L}"
    [[ -n "${CLUSTER_CHI:-}" ]] && CMD_ARGS+=" --chi=${CLUSTER_CHI}"
    [[ -n "${CLUSTER_FILLING:-}" ]] && CMD_ARGS+=" --set_filling=${CLUSTER_FILLING}"
    [[ -n "${CLUSTER_SIZES:-}" ]] && CMD_ARGS+=" --cluster_sizes=${CLUSTER_SIZES}"
    [[ -n "${CLUSTER_SC_SIZES:-}" ]] && CMD_ARGS+=" --supercluster_sizes=${CLUSTER_SC_SIZES}"
    [[ -n "${CLUSTER_U_VALUES:-}" ]] && CMD_ARGS+=" --U_values=${CLUSTER_U_VALUES}"
    [[ -n "${CLUSTER_V_VALUES:-}" ]] && CMD_ARGS+=" --V_values=${CLUSTER_V_VALUES}"
    [[ -n "${CLUSTER_RESULTS:-}" ]] && CMD_ARGS+=" --results=${CLUSTER_RESULTS}"

    # If still just the function name with no overrides, use defaults
    if [[ "$CMD_ARGS" == "filling_cluster_sizes" && -z "${CLUSTER_L:-}" ]]; then
        CMD_ARGS="$DEFAULT_CMD"
    fi
fi

echo ""
echo "============================================================"
echo "Running cluster comparison"
echo "============================================================"
echo "Command: python -m aah_code.cluster_model.cluster_runner $CMD_ARGS"
echo "============================================================"
echo ""

# Run the command
python -m aah_code.cluster_model.cluster_runner $CMD_ARGS

echo ""
echo "Job completed at $(date)"
