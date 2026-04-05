#!/bin/bash
# =============================================================================
# Submit a cluster job from a YAML config file
# =============================================================================
#
# This script reads both SLURM settings and Python parameters from a single
# YAML config file, generates an sbatch script, syncs to the cluster, and submits.
#
# Supports two modes:
#   - Single job (default): Runs entire computation on one node
#   - Array job (array: true): Parallelizes across U/V grid, then merges
#
# Usage:
#   ./scripts/submit_from_yaml.sh configs/my_run.yaml
#   ./scripts/submit_from_yaml.sh configs/my_run.yaml --watch
#
# =============================================================================

set -euo pipefail

WATCH_AFTER=false

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch|-w)
            WATCH_AFTER=true
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [--watch]"
    echo ""
    echo "Options:"
    echo "  --watch, -w    Start syncing results after job is submitted"
    exit 1
fi

CONFIG_YAML="$1"

if [[ ! -f "$CONFIG_YAML" ]]; then
    echo "Error: Config file not found: $CONFIG_YAML"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Parse YAML and extract all settings using Python
YAML_SETTINGS=$(python3 - "$CONFIG_YAML" <<'PY'
import sys
import yaml
import json
import re
from pathlib import Path

config_path = Path(sys.argv[1])
with open(config_path) as f:
    config = yaml.safe_load(f)

# Variable substitution: replace ${var} with actual values from config
def substitute_vars(s, variables):
    """Substitute ${var} patterns in string with actual config values."""
    if not isinstance(s, str):
        return s
    def replacer(match):
        var_name = match.group(1)
        if var_name in variables:
            return str(variables[var_name])
        return match.group(0)  # Keep original if not found
    return re.sub(r'\$\{(\w+)\}', replacer, s)

slurm = config.get('slurm', {})

# SLURM defaults
defaults = {
    'job_name': 'cluster_run',
    'time': '08:00:00',
    'mem': '32G',
    'partition': 'eng-research-gpu',
    'account': 'bbradlyn-ic',
    'nodes': 1,
    'ntasks': 1,
    'ntasks_per_node': 1,
    'gres': 'gpu:1',
    'mail_user': '',
    'mail_type': 'ALL',
    'conda_env': '',
    'use_venv': True,
    'exclusive': False,
    'array_max_concurrent': 0,  # 0 = unlimited, >0 = max concurrent tasks
    'separate_jobs': False,  # Submit as individual jobs instead of array
}

for key, default in defaults.items():
    value = slurm.get(key, default)
    if isinstance(value, bool):
        value = 'true' if value else 'false'
    # For array_max_concurrent, only set if > 0 (0 means unlimited)
    if key == 'array_max_concurrent' and int(value) <= 0:
        value = ''
    print(f"SLURM_{key.upper()}='{value}'")

# Config settings - apply variable substitution
output_dir = substitute_vars(config.get('output_dir', ''), config)
print(f"CONFIG_OUTPUT_DIR='{output_dir}'")

watch = config.get('watch', False)
print(f"CONFIG_WATCH={'true' if watch else 'false'}")

array_mode = config.get('array', False)
print(f"CONFIG_ARRAY={'true' if array_mode else 'false'}")

command = config.get('command', 'filling_cluster_sizes')
print(f"CONFIG_COMMAND='{command}'")

# Export full config as JSON for manifest generation
print(f"CONFIG_JSON='{json.dumps(config)}'")
PY
)

eval "$YAML_SETTINGS"

# Get paths
CONFIG_BASENAME=$(basename "$CONFIG_YAML")
CONFIG_RELPATH=$(python3 -c "import os; print(os.path.relpath('$CONFIG_YAML', '$PROJECT_ROOT'))")

echo "============================================================"
echo "Submitting job from YAML config"
echo "============================================================"
echo "Config:     $CONFIG_YAML"
echo "Job name:   $SLURM_JOB_NAME"
echo "Command:    $CONFIG_COMMAND"
echo "Array mode: $CONFIG_ARRAY"
echo "Time:       $SLURM_TIME"
echo "Memory:     $SLURM_MEM"
echo "Partition:  $SLURM_PARTITION"
echo "GPU:        $SLURM_GRES"
echo "============================================================"

# -----------------------------------------------------------------------------
# Array job mode
# -----------------------------------------------------------------------------
if [[ "$CONFIG_ARRAY" == "true" ]]; then
    echo ""
    echo "Setting up array job..."

    # Create timestamped run folder
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_ID="${CONFIG_OUTPUT_DIR:-run}_${TIMESTAMP}"
    PARTIAL_DIR="$PROJECT_ROOT/aah/aah_code/cluster_model/large_files/partials/${RUN_ID}"
    MANIFEST_PATH="$PARTIAL_DIR/manifest.json"

    mkdir -p "$PARTIAL_DIR"
    echo "Run folder: $PARTIAL_DIR"

    # Generate manifest
    python3 - "$CONFIG_JSON" "$MANIFEST_PATH" <<'PY'
import sys
import json
from pathlib import Path

config = json.loads(sys.argv[1])
manifest_path = Path(sys.argv[2])

U_values = config.get('U_values', [0, 1, 2, 3, 4, 5, 7, 10, 20, 30])

# Check if V is a single value or a list
# filling_with_int_cluster_sizes uses single V, others use V_values list
if 'V' in config and not isinstance(config['V'], list):
    # Single V value (e.g., filling_with_int_cluster_sizes)
    V_values = [config['V']]
else:
    V_values = config.get('V_values', [1e-6, 1, 2, 3, 5])

manifest = []
for U in U_values:
    for V in V_values:
        task = dict(config)  # Copy all config
        task['U'] = U
        task['V'] = V
        manifest.append(task)

manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"Created manifest with {len(manifest)} tasks")
PY

    TASK_COUNT=$(python3 -c "import json; print(len(json.loads(open('$MANIFEST_PATH').read())))")
    echo "Total tasks: $TASK_COUNT"

    if [[ "$TASK_COUNT" -lt 1 ]]; then
        echo "Error: No tasks in manifest"
        exit 1
    fi

    # Generate array sbatch script
    # Use relative path for SLURM output (relative to SLURM_SUBMIT_DIR on cluster)
    RELATIVE_PARTIAL_DIR="aah/aah_code/cluster_model/large_files/partials/${RUN_ID}"
    ARRAY_SBATCH="$PARTIAL_DIR/run_array.sbatch"

    # Generate sbatch template WITHOUT --array, --output, --error
    # These are added at submission time to support both array and separate_jobs modes
    cat > "$ARRAY_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks=${SLURM_NTASKS}
#SBATCH --ntasks-per-node=${SLURM_NTASKS_PER_NODE}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
# NOTE: --array, --output, --error are added at submission time
SBATCH_EOF

    [[ -n "$SLURM_GRES" ]] && echo "#SBATCH --gres=${SLURM_GRES}" >> "$ARRAY_SBATCH"
    [[ -n "$SLURM_MAIL_USER" ]] && echo "#SBATCH --mail-user=${SLURM_MAIL_USER}" >> "$ARRAY_SBATCH"
    [[ "$SLURM_EXCLUSIVE" == "true" ]] && echo "#SBATCH --exclusive" >> "$ARRAY_SBATCH"

    cat >> "$ARRAY_SBATCH" <<'SBATCH_ENV'

set -euo pipefail

# Ensure environment modules are available
if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
fi

# Load python module (not anaconda to avoid PYTHONPATH pollution)
module load python/3.11
module load cuda/12.6
module load intel/mkl/latest

# Disable GPU eigensolver (CuPy fallback causes hangs) and pin BLAS threads
export CUDA_VISIBLE_DEVICES=-1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Enable numba caching to avoid recompilation (QuSpin uses numba)
export NUMBA_CACHE_DIR="$SLURM_SUBMIT_DIR/.numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

# Clear PYTHONPATH to avoid conflicts with venv
unset PYTHONPATH

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SBATCH_ENV

    # Add venv setup with locking for array jobs
    cat >> "$ARRAY_SBATCH" <<VENV_SETUP

VENV_DIR="\$SLURM_SUBMIT_DIR/.venv"
LOCKFILE="\$SLURM_SUBMIT_DIR/.venv.lock"

need_setup=0
if [[ ! -x "\$VENV_DIR/bin/python" ]]; then
    need_setup=1
elif [[ ! -f "\$VENV_DIR/.deps_installed" ]]; then
    need_setup=1
fi

if [[ "\$need_setup" -eq 1 ]]; then
    {
        flock 9
        if [[ ! -x "\$VENV_DIR/bin/python" ]]; then
            python3 -m venv "\$VENV_DIR"
        fi
        source "\$VENV_DIR/bin/activate"
        pip install --upgrade pip setuptools wheel
        pushd "\$SLURM_SUBMIT_DIR/aah" >/dev/null
        pip install -e .
        popd >/dev/null
        touch "\$VENV_DIR/.deps_installed"
    } 9>"\$LOCKFILE"
fi

source "\$VENV_DIR/bin/activate"

# Extract task parameters (works for both array and separate_jobs modes)
TASK_ID=\${SLURM_ARRAY_TASK_ID:-\${TASK_ID:-0}}
MANIFEST_PATH="\$SLURM_SUBMIT_DIR/${RELATIVE_PARTIAL_DIR}/manifest.json"
PARTIAL_DIR="\$SLURM_SUBMIT_DIR/${RELATIVE_PARTIAL_DIR}"

PARAMS_FILE="\$PARTIAL_DIR/params_\${TASK_ID}.json"
OUT_PATH="\$PARTIAL_DIR/partial_\${TASK_ID}.pkl"

export MANIFEST_PATH PARTIAL_DIR TASK_ID

echo "DEBUG: TASK_ID=\$TASK_ID, SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-unset}"

python3 - "\$TASK_ID" <<'EXTRACT_PY'
import json, sys
from pathlib import Path
import os

manifest_path = Path(os.environ["MANIFEST_PATH"])
partial_dir = Path(os.environ["PARTIAL_DIR"])

# Get task_id from command line argument (most reliable)
task_id = int(sys.argv[1])
print(f"Extracting params for task {task_id}")

entries = json.loads(manifest_path.read_text())
if task_id >= len(entries):
    raise SystemExit(f"Task ID {task_id} out of range (manifest has {len(entries)} entries)")

params = entries[task_id]
params_file = partial_dir / f"params_{task_id}.json"
params_file.write_text(json.dumps(params))
print(f"Wrote params for task {task_id} to {params_file}")
EXTRACT_PY

# Run single task
python -m aah_code.cluster_model.run_single_comparison \\
    --params "\$PARAMS_FILE" \\
    --output "\$OUT_PATH"

echo "Task \$TASK_ID completed at \$(date)"
VENV_SETUP

    mkdir -p "$PARTIAL_DIR/logs"

    # Generate merge sbatch script
    # Use relative paths for cluster
    RELATIVE_MERGE_OUTPUT="${RELATIVE_PARTIAL_DIR}/merged_${TIMESTAMP}.pkl"
    MERGE_SBATCH="$PARTIAL_DIR/run_merge.sbatch"

    cat > "$MERGE_SBATCH" <<MERGE_EOF
#!/bin/bash
#SBATCH --job-name=${SLURM_JOB_NAME}_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --output=${RELATIVE_PARTIAL_DIR}/logs/merge_%j.out
#SBATCH --error=${RELATIVE_PARTIAL_DIR}/logs/merge_%j.err

set -euo pipefail

# Ensure environment modules are available
if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
fi

# Load python module (not anaconda to avoid PYTHONPATH pollution)
module load python/3.11

# Clear PYTHONPATH to avoid conflicts with venv
unset PYTHONPATH

cd "\$SLURM_SUBMIT_DIR"

VENV_DIR="\$SLURM_SUBMIT_DIR/.venv"
if [[ -x "\$VENV_DIR/bin/python" ]]; then
    source "\$VENV_DIR/bin/activate"
fi

PARTIAL_DIR="\$SLURM_SUBMIT_DIR/${RELATIVE_PARTIAL_DIR}"

echo "Merging partial results..."
python -m aah_code.cluster_model.merge_comparison_results \\
    --partials "\$PARTIAL_DIR/partial_*.pkl" \\
    --output "\$PARTIAL_DIR/merged_${TIMESTAMP}.pkl"

echo "Merge completed at \$(date)"
echo "Output: \$PARTIAL_DIR/merged_${TIMESTAMP}.pkl"
MERGE_EOF

    # Copy scripts to project for syncing
    cp "$ARRAY_SBATCH" "$PROJECT_ROOT/aah/aah_code/cluster_model/scripts/slurm/generated_array.sbatch"
    cp "$MERGE_SBATCH" "$PROJECT_ROOT/aah/aah_code/cluster_model/scripts/slurm/generated_merge.sbatch"

    # Sync to cluster (reuse SSH connection logic from submit_cluster_job.sh)
    CONFIG_FILE="${CLUSTER_CONFIG_TOML:-$PROJECT_ROOT/aah/pyproject.toml}"

    eval "$(python3 - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

def load_toml(path):
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except:
        pass
    try:
        import tomli
        with open(path, "rb") as f:
            return tomli.load(f)
    except:
        return {}

config_path = Path(sys.argv[1])
if not config_path.exists():
    sys.exit(0)

cfg = load_toml(config_path)
cluster = cfg.get("tool", {}).get("cluster", {})
remote = cluster.get("remote", {})

print(f"REMOTE_USER='{remote.get('user', 'dmitry2')}'")
print(f"REMOTE_HOST='{remote.get('host', 'cc-login.campuscluster.illinois.edu')}'")
print(f"REMOTE_PROJECT_ROOT='{remote.get('project_root', '/projects/illinois/eng/physics/bbradlyn/dmitry2/k_blocking/new_code')}'")
PY
)"

    SSH_CONTROL_DIR="$HOME/.ssh/cm"
    SSH_CONTROL_PATH="$SSH_CONTROL_DIR/%r@%h:%p"
    SSH_TARGET="$REMOTE_USER@$REMOTE_HOST"
    mkdir -p "$SSH_CONTROL_DIR"

    # Establish SSH connection if needed
    if ! ssh -o "ControlPath=$SSH_CONTROL_PATH" -O check "$SSH_TARGET" 2>/dev/null; then
        echo "Establishing SSH connection (2FA may be required)..."
        ssh -o "ControlMaster=auto" -o "ControlPath=$SSH_CONTROL_PATH" -o "ControlPersist=yes" -fN "$SSH_TARGET"
    fi

    echo ""
    echo "Syncing project to cluster..."
    rsync -az --delete \
        --exclude=".git/" \
        --exclude="large_files/" \
        --exclude=".venv/" \
        --exclude="__pycache__/" \
        --exclude="*.pyc" \
        --exclude="*.pyo" \
        --exclude=".pytest_cache/" \
        --exclude="*.egg-info/" \
        --exclude=".mypy_cache/" \
        --exclude=".ruff_cache/" \
        --exclude="wandb/" \
        --exclude="logs/" \
        --exclude=".ipynb_checkpoints/" \
        --exclude="*.pkl" \
        --exclude="*.h5" \
        --exclude="*.hdf5" \
        -e "ssh -o ControlPath=$SSH_CONTROL_PATH" \
        "$PROJECT_ROOT/" "$SSH_TARGET:$REMOTE_PROJECT_ROOT/"

    # Create remote partials directory structure
    REMOTE_PARTIAL_DIR="$REMOTE_PROJECT_ROOT/aah/aah_code/cluster_model/large_files/partials/${RUN_ID}"
    ssh -o "ControlPath=$SSH_CONTROL_PATH" "$SSH_TARGET" "mkdir -p '$REMOTE_PARTIAL_DIR/logs'"

    # Sync the partials directory (manifest, scripts)
    rsync -az \
        -e "ssh -o ControlPath=$SSH_CONTROL_PATH" \
        "$PARTIAL_DIR/" "$SSH_TARGET:$REMOTE_PARTIAL_DIR/"

    echo ""
    REMOTE_TASK_SCRIPT="$REMOTE_PARTIAL_DIR/run_array.sbatch"
    REMOTE_MERGE_SCRIPT="$REMOTE_PARTIAL_DIR/run_merge.sbatch"
    LOGS_DIR="${RELATIVE_PARTIAL_DIR}/logs"

    # -------------------------------------------------------------------------
    # Separate jobs mode: submit each task as independent job
    # -------------------------------------------------------------------------
    if [[ "$SLURM_SEPARATE_JOBS" == "true" ]]; then
        echo "Submitting ${TASK_COUNT} separate jobs..."

        JOB_IDS=()
        for i in $(seq 0 $((TASK_COUNT-1))); do
            JOB_ID=$(ssh -o "ControlPath=$SSH_CONTROL_PATH" "$SSH_TARGET" bash <<EOF
set -euo pipefail
cd "$REMOTE_PROJECT_ROOT"
sbatch --parsable \
    --export=ALL,TASK_ID=$i \
    --output="${LOGS_DIR}/%x_task${i}_%j.out" \
    --error="${LOGS_DIR}/%x_task${i}_%j.err" \
    "$REMOTE_TASK_SCRIPT"
EOF
)
            JOB_IDS+=("$JOB_ID")
            echo "  Task $i: job $JOB_ID"
        done

        # Build dependency string (afterok:123:456:789:...)
        DEP_STRING=$(IFS=:; echo "${JOB_IDS[*]}")

        echo ""
        echo "Submitting merge job (depends on all ${TASK_COUNT} tasks)..."

        MERGE_JOB_ID=$(ssh -o "ControlPath=$SSH_CONTROL_PATH" "$SSH_TARGET" bash <<EOF
set -euo pipefail
cd "$REMOTE_PROJECT_ROOT"
sbatch --parsable --dependency=afterok:${DEP_STRING} "$REMOTE_MERGE_SCRIPT"
EOF
)

        echo "Merge job submitted: $MERGE_JOB_ID"
        echo ""
        echo "============================================================"
        echo "Separate jobs submitted:"
        echo "  Tasks: ${JOB_IDS[*]}"
        echo "  Merge: $MERGE_JOB_ID (depends on all tasks)"
        echo ""
        echo "Output will be: ${RELATIVE_PARTIAL_DIR}/merged_${TIMESTAMP}.pkl"
        echo "============================================================"

    # -------------------------------------------------------------------------
    # Array job mode (default): submit as single array job
    # -------------------------------------------------------------------------
    else
        echo "Submitting array job..."

        # Build array spec with optional throttling
        ARRAY_SPEC="0-$((TASK_COUNT-1))"
        [[ -n "$SLURM_ARRAY_MAX_CONCURRENT" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${SLURM_ARRAY_MAX_CONCURRENT}"

        ARRAY_JOB_ID=$(ssh -o "ControlPath=$SSH_CONTROL_PATH" "$SSH_TARGET" bash <<EOF
set -euo pipefail
cd "$REMOTE_PROJECT_ROOT"
sbatch --parsable \
    --array=${ARRAY_SPEC} \
    --output="${LOGS_DIR}/%x_%A_%a.out" \
    --error="${LOGS_DIR}/%x_%A_%a.err" \
    "$REMOTE_TASK_SCRIPT"
EOF
)

        echo "Array job submitted: $ARRAY_JOB_ID"
        echo ""
        echo "Submitting merge job (will run after array completes)..."

        MERGE_JOB_ID=$(ssh -o "ControlPath=$SSH_CONTROL_PATH" "$SSH_TARGET" bash <<EOF
set -euo pipefail
cd "$REMOTE_PROJECT_ROOT"
sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} "$REMOTE_MERGE_SCRIPT"
EOF
)

        echo "Merge job submitted: $MERGE_JOB_ID (depends on $ARRAY_JOB_ID)"
        echo ""
        echo "============================================================"
        echo "Job chain submitted:"
        echo "  Array: $ARRAY_JOB_ID (${TASK_COUNT} tasks)"
        echo "  Merge: $MERGE_JOB_ID (runs after array completes)"
        echo ""
        echo "Output will be: ${RELATIVE_PARTIAL_DIR}/merged_${TIMESTAMP}.pkl"
        echo "============================================================"
    fi

    # Update output_dir for watcher to use run folder
    CONFIG_OUTPUT_DIR="$RUN_ID"

# -----------------------------------------------------------------------------
# Single job mode (original behavior)
# -----------------------------------------------------------------------------
else
    # Generate single-job sbatch script
    GENERATED_SBATCH=$(mktemp /tmp/run_XXXXXX.sbatch)

    cat > "$GENERATED_SBATCH" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks=${SLURM_NTASKS}
#SBATCH --ntasks-per-node=${SLURM_NTASKS_PER_NODE}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
SBATCH_EOF

    [[ -n "$SLURM_GRES" ]] && echo "#SBATCH --gres=${SLURM_GRES}" >> "$GENERATED_SBATCH"
    [[ -n "$SLURM_MAIL_USER" ]] && echo "#SBATCH --mail-user=${SLURM_MAIL_USER}" >> "$GENERATED_SBATCH"
    [[ -n "$SLURM_MAIL_USER" ]] && echo "#SBATCH --mail-type=${SLURM_MAIL_TYPE}" >> "$GENERATED_SBATCH"

    cat >> "$GENERATED_SBATCH" <<SBATCH_ENV

# Ensure environment modules are available
if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
fi

# Load python module (not anaconda to avoid PYTHONPATH pollution)
module load python/3.11
module load cuda/12.6
module list

# Clear PYTHONPATH to avoid conflicts with venv
unset PYTHONPATH

echo "Running on \$(hostname)"
nvidia-smi || true

cd "\$SLURM_SUBMIT_DIR"
mkdir -p logs

SBATCH_ENV

    if [[ -n "$SLURM_CONDA_ENV" ]]; then
        cat >> "$GENERATED_SBATCH" <<CONDA_SETUP
# Activate existing conda environment
echo "Activating conda environment: ${SLURM_CONDA_ENV}"
source activate ${SLURM_CONDA_ENV}

pushd "\$SLURM_SUBMIT_DIR/aah" >/dev/null
pip install -e . --quiet
popd >/dev/null

CONDA_SETUP
    else
        cat >> "$GENERATED_SBATCH" <<'VENV_SETUP'
# Python setup with venv
pick_python() {
    for bin in python3.12 python3.11 python3; do
        if command -v "$bin" >/dev/null 2>&1; then
            if "$bin" -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
                echo "$bin"
                return
            fi
        fi
    done
}

PYTHON_BIN="$(pick_python)"
[[ -z "$PYTHON_BIN" ]] && { echo "No Python >=3.11 found" >&2; exit 1; }

VENV_DIR="$SLURM_SUBMIT_DIR/.venv"
[[ ! -d "$VENV_DIR" ]] && "$PYTHON_BIN" -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

pushd "$SLURM_SUBMIT_DIR/aah" >/dev/null
pip install -e .
popd >/dev/null

# Skip CuPy — GPU eigensolver causes convergence failures and fallback overhead.
# All computation uses CPU (scipy/numpy). See used_data.md for details.
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

VENV_SETUP
    fi

    echo "" >> "$GENERATED_SBATCH"
    echo "# Run from config" >> "$GENERATED_SBATCH"
    echo "python -m aah_code.cluster_model.cluster_runner --config=\"\${SLURM_SUBMIT_DIR}/${CONFIG_RELPATH}\"" >> "$GENERATED_SBATCH"
    echo "" >> "$GENERATED_SBATCH"
    echo "echo \"Job completed at \$(date)\"" >> "$GENERATED_SBATCH"

    echo ""
    echo "Generated sbatch script: $GENERATED_SBATCH"
    echo ""

    FINAL_SBATCH="$PROJECT_ROOT/aah/aah_code/cluster_model/scripts/slurm/generated_run.sbatch"
    cp "$GENERATED_SBATCH" "$FINAL_SBATCH"

    export SLURM_SCRIPT_RELATIVE="aah/aah_code/cluster_model/scripts/slurm/generated_run.sbatch"
    "$SCRIPT_DIR/submit_cluster_job.sh"
fi

# Optionally start watching for results
if $WATCH_AFTER || [[ "$CONFIG_WATCH" == "true" ]]; then
    echo ""
    echo "============================================================"
    echo "Starting result watcher..."
    echo "============================================================"

    if [[ -n "$CONFIG_OUTPUT_DIR" ]]; then
        WATCH_CMD="'$SCRIPT_DIR/sync_results.sh' --dir '$CONFIG_OUTPUT_DIR' --watch"
    else
        WATCH_CMD="'$SCRIPT_DIR/sync_results.sh' --today --watch"
    fi

    TMUX_SESSION="watch_${SLURM_JOB_NAME}"

    if command -v tmux &>/dev/null; then
        # Kill any existing session with the same name
        tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
        tmux new-session -d -s "$TMUX_SESSION" "$WATCH_CMD"
        echo "Watcher running in tmux session: $TMUX_SESSION"
        echo "  Attach:  tmux attach -t $TMUX_SESSION"
        echo "  Kill:    tmux kill-session -t $TMUX_SESSION"
    else
        echo "tmux not found, running watcher in foreground..."
        eval exec "$WATCH_CMD"
    fi
fi
