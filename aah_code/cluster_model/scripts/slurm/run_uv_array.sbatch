#!/bin/bash
#SBATCH --job-name=gpu_9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=IllinoisComputes
#SBATCH --account=bbradlyn-ic
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-1
#SBATCH --begin=now
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err


# Array index should be provided at submission time, e.g. --array=0-9

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

MANIFEST_PATH="${MANIFEST_PATH:-$PROJECT_ROOT/large_files/manifest_uv.json}"
PARTIAL_DIR="${PARTIAL_DIR:-$PROJECT_ROOT/large_files/partials_uv}"

module load cuda/12.6

# Ensure environment modules are available in non-interactive shells.
if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
fi

module load anaconda3/2024.10
module list

cd "$PROJECT_ROOT"
mkdir -p logs "$PARTIAL_DIR"

PYTHON_BIN=python3
# Use a clean project-local venv for Python deps (including quspin).
VENV_DIR="$PROJECT_ROOT/.venv"
LOCKFILE="$PROJECT_ROOT/.venv.lock"

need_setup=0
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    need_setup=1
elif [[ ! -f "$VENV_DIR/.deps_installed" ]]; then
    need_setup=1
elif ! "$VENV_DIR/bin/python" -c "import matplotlib" >/dev/null 2>&1; then
    need_setup=1
fi

# Create venv and install dependencies once, under a lock to avoid races across array tasks.
# Avoid deleting an in-use venv on NFS (can cause "Directory not empty" errors).
if [[ "$need_setup" -eq 1 ]]; then
    {
        flock 9

        need_setup_locked=0
        if [[ ! -x "$VENV_DIR/bin/python" ]]; then
            need_setup_locked=1
        elif [[ ! -f "$VENV_DIR/.deps_installed" ]]; then
            need_setup_locked=1
        elif ! "$VENV_DIR/bin/python" -c "import matplotlib" >/dev/null 2>&1; then
            need_setup_locked=1
        fi

        if [[ "$need_setup_locked" -eq 1 ]]; then
            if [[ ! -x "$VENV_DIR/bin/python" ]]; then
                if [[ -d "$VENV_DIR" ]]; then
                    BACKUP_DIR="${VENV_DIR}.old.${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_JOB_ID:-0}.${SLURM_ARRAY_TASK_ID:-0}.$(date +%s)"
                    echo "Existing venv at $VENV_DIR looks invalid; moving aside to $BACKUP_DIR"
                    mv "$VENV_DIR" "$BACKUP_DIR"
                    rm -rf "$BACKUP_DIR" 2>/dev/null || echo "Warning: could not fully remove $BACKUP_DIR (likely NFS .nfs files); leaving it."
                fi
                echo "Creating virtual environment at $VENV_DIR"
                "$PYTHON_BIN" -m venv "$VENV_DIR"
            fi

            source "$VENV_DIR/bin/activate"
            "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
            pushd "$PROJECT_ROOT/aah" >/dev/null
            "$VENV_DIR/bin/python" -m pip install -e .
            popd >/dev/null
            "$VENV_DIR/bin/python" -m pip install matplotlib
            touch "$VENV_DIR/.deps_installed"
        fi
    } 9>"$LOCKFILE"
fi

source "$VENV_DIR/bin/activate"

export TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export TIMING_CSV="${TIMING_CSV:-$PARTIAL_DIR/timings_task${TASK_ID}.csv}"

read PARAMS_FILE OUT_PATH <<EOF
$("$VENV_DIR/bin/python" - <<'PY'
import json, os
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST_PATH"])
partial_dir = Path(os.environ["PARTIAL_DIR"])
task_id = int(os.environ.get("TASK_ID", "0"))

if not manifest_path.exists():
    raise SystemExit(f"Manifest not found: {manifest_path}")

entries = json.loads(manifest_path.read_text())
if task_id < 0 or task_id >= len(entries):
    raise SystemExit(f"TASK_ID {task_id} out of range (0..{len(entries)-1})")

params = entries[task_id]
out_path = partial_dir / f"partial_u{params['U']}_v{params['V']}.pkl"
params_file = partial_dir / f"params_{task_id}.json"
params_file.write_text(json.dumps(params))

print(f"{params_file} {out_path}")
PY
)
EOF

"$VENV_DIR/bin/python" "$PROJECT_ROOT/aah/aah_code/cluster_model/run_single_uv.py" \
  --params "$PARAMS_FILE" \
  --output "$OUT_PATH"
