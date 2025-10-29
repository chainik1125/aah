#!/bin/bash
#SBATCH --job-name=aah_fac_convergence_9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=IllinoisComputes
#SBATCH --account=bbradlyn-ic
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-1
#SBATCH --begin=now
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Optional runtime configuration (modules, conda env, command) loaded from pyproject.toml.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CLUSTER_RUN_CONFIG:-$SCRIPT_DIR/../../../../pyproject.toml}"

DEFAULT_RUN_COMMAND="python -m aah_code.cluster_model.main"
RUN_COMMAND="$DEFAULT_RUN_COMMAND"
RUN_WORKDIR="$SLURM_SUBMIT_DIR"
RUN_ACTIVATE=""
RUN_MODULES=()
PYTHON_CANDIDATE="${CLUSTER_PYTHON:-}"

# Ensure environment modules are available in non-interactive shells.
if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
fi

PYTHON_FOR_CONFIG="${PYTHON_FOR_CONFIG:-python3}"
if ! command -v "$PYTHON_FOR_CONFIG" >/dev/null 2>&1; then
    PYTHON_FOR_CONFIG=python
fi

if [[ -f "$CONFIG_FILE" ]]; then
    eval "$("$PYTHON_FOR_CONFIG" - <<'PY'
import ast
import os
import shlex
import sys
from pathlib import Path

def load_toml(path: Path):
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:
        with path.open("rb") as fh:
            return tomllib.load(fh)

    try:
        import tomli  # type: ignore
    except ModuleNotFoundError:
        tomli = None

    if tomli is not None:
        with path.open("rb") as fh:
            return tomli.load(fh)

    return load_simple_toml(path)

def parse_value(value: str):
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip().strip('"').strip("'")

def load_simple_toml(path: Path):
    data = {}
    current = data
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                current = data
                for part in section.split("."):
                    current = current.setdefault(part, {})
                continue
            if "=" in line:
                key, value = [segment.strip() for segment in line.split("=", 1)]
                current[key] = parse_value(value)
    return data

cfg_path = Path(os.environ["CONFIG_FILE"])
if not cfg_path.exists():
    sys.exit(0)

cfg = load_toml(cfg_path)

cluster_cfg = cfg.get("tool", {}).get("cluster", {})

env_cfg = cluster_cfg.get("environment", {})
run_cfg = cluster_cfg.get("run", {})

modules = env_cfg.get("modules", [])
activate = env_cfg.get("activate", "")
env_vars = env_cfg.get("variables", {})

command = run_cfg.get("command", "")
args = run_cfg.get("args", [])
working_dir = run_cfg.get("working_dir", "")

def shell_array(name, values):
    elems = " ".join(shlex.quote(str(v)) for v in values)
    print(f"{name}=({elems})")

shell_array("RUN_MODULES", modules)
print(f"RUN_ACTIVATE={shlex.quote(str(activate))}")
print(f"RUN_WORKDIR={shlex.quote(str(working_dir))}")

cmd_parts = []
if command:
    cmd_parts.append(command)
if isinstance(args, list) and args:
    cmd_parts.extend(shlex.quote(str(a)) for a in args)
final_cmd = " ".join(cmd_parts) if cmd_parts else ""
if final_cmd:
    print(f"RUN_COMMAND={shlex.quote(final_cmd)}")

for key, value in env_vars.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"
fi

if ((${#RUN_MODULES[@]})); then
    for mod in "${RUN_MODULES[@]}"; do
        module load "$mod"
    done
else
    module load anaconda3/2024.10
fi
module list

if [[ -n "$RUN_ACTIVATE" ]]; then
    eval "$RUN_ACTIVATE"
fi

if [[ -n "$RUN_WORKDIR" ]]; then
    if [[ "$RUN_WORKDIR" = /* ]]; then
        cd "$RUN_WORKDIR"
    else
        cd "$SLURM_SUBMIT_DIR/$RUN_WORKDIR"
    fi
else
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p logs

pick_python() {
    local raw_candidates=("$PYTHON_CANDIDATE" python3.12 python3.11 python3 python)
    local candidates=()
    local seen=""
    for bin in "${raw_candidates[@]}"; do
        [[ -z "$bin" ]] && continue
        if [[ " $seen " != *" $bin "* ]]; then
            candidates+=("$bin")
            seen+=" $bin"
        fi
    done
    for bin in "${candidates[@]}"; do
        if command -v "$bin" >/dev/null 2>&1; then
            if "$bin" - <<'PY' >/dev/null 2>&1
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major > 3 or (major == 3 and minor >= 11)) else 1)
PY
            then
                echo "$bin"
                return
            fi
        fi
    done
    echo ""
}

PYTHON_BIN="$(pick_python)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "No Python >=3.11 found (tried $PYTHON_CANDIDATE, python3.12, python3.11, python3)." >&2
    exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"

VENV_DIR="$SLURM_SUBMIT_DIR/.venv"

needs_recreate=0
if [[ -d "$VENV_DIR" ]]; then
    if ! "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major > 3 or (major == 3 and minor >= 11)) else 1)
PY
    then
        echo "Existing venv uses an older Python; recreating..."
        needs_recreate=1
    fi
fi

if [[ $needs_recreate -eq 1 ]]; then
    rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR using $PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

pushd "$SLURM_SUBMIT_DIR/aah" >/dev/null
if ! "$VENV_DIR/bin/python" -m pip install -e ".[dev]" >/dev/null 2>&1; then
    echo "Warning: installing dev extras failed, falling back to base install."
    "$VENV_DIR/bin/python" -m pip install -e .
fi
popd >/dev/null

if [[ "$RUN_COMMAND" == "$DEFAULT_RUN_COMMAND" ]]; then
    RUN_COMMAND="$VENV_DIR/bin/python -m aah_code.cluster_model.main"
fi

echo "Running convergence study on $(hostname)"
if [[ -n "$RUN_COMMAND" ]]; then
    eval "$RUN_COMMAND"
else
    "$VENV_DIR/bin/python" -m aah_code.cluster_model.main
fi
