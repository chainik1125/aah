#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=eng-research-gpu
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

module load cuda/12.6
nvcc --version
nvidia-smi

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
"$VENV_DIR/bin/python" -m pip install -e .
popd >/dev/null

if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
        if ! "$VENV_DIR/bin/python" -c "import cupy" >/dev/null 2>&1; then
            echo "GPU detected; installing CUDA extras"
            "$VENV_DIR/bin/python" -m pip install -e .[cuda]
        else
            echo "CuPy already available in venv; skipping install."
        fi
    fi
fi

# ------------------------------------------------------------
# Parameter grid for timing/k-point CSV
# ------------------------------------------------------------
export OUT_CSV="kpoint_benchmark.csv"
export U_LIST="0.5 1.0"
export V_LIST="0.1 1.0"
export T_LIST="1.0"
export NC_LIST="2 4 8"
export INT_SEP_LIST="1,2 1,4 1,8"
export V_SEP="1,2"
export L=24
export CHI=32
export SOLVER_METHOD="sparse_ED"
export STATES_RETAINED=6

echo "cluster_size,super_cluster_size,int_sep,v_sep,U,V,t,method,K_points" > "$OUT_CSV"

"$VENV_DIR/bin/python" - <<'PY'
import csv, os, numpy as np

OUT_CSV = os.environ["OUT_CSV"]
U_LIST       = os.environ["U_LIST"].split()
V_LIST       = os.environ["V_LIST"].split()
T_LIST       = os.environ["T_LIST"].split()
NC_LIST      = os.environ["NC_LIST"].split()
INT_SEP_LIST = os.environ["INT_SEP_LIST"].split()
V_SEP        = os.environ.get("V_SEP", "1,2")
L            = int(os.environ.get("L", 24))
CHI          = int(os.environ.get("CHI", 32))
SOLVER_METHOD= os.environ.get("SOLVER_METHOD", "sparse_ED")
STATES       = int(os.environ.get("STATES_RETAINED", 6))

from aah_code.cluster_model.clustering import generate_clusters, convert_site_clusters_to_k
from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations
from aah_code.real_space_dmrg import run_dmrg_method, get_gnd

def parse_ratio(txt):
    p, q = txt.split(",")
    return (int(p), int(q))

v_sep_ratio = parse_ratio(V_SEP)

def row(writer, Nc, sc_size, int_sep, U, V, t, method, kpts):
    writer.writerow({
        "cluster_size": Nc,
        "super_cluster_size": sc_size,
        "int_sep": f"{int_sep[0]}/{int_sep[1]}",
        "v_sep": f"{v_sep_ratio[0]}/{v_sep_ratio[1]}",
        "U": U,
        "V": V,
        "t": t,
        "method": method,
        "K_points": kpts,
    })

with open(OUT_CSV, "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=[
        "cluster_size","super_cluster_size","int_sep","v_sep",
        "U","V","t","method","K_points"
    ])
    for U in map(float, U_LIST):
        mu_0 = U / 2.0
        for V in map(float, V_LIST):
            for t in map(float, T_LIST):
                for Nc in map(int, NC_LIST):
                    for int_sep_txt in INT_SEP_LIST:
                        try:
                            int_sep = parse_ratio(int_sep_txt)
                        except Exception as exc:
                            row(writer, Nc, "N/A", ("N/A","N/A"), U, V, t, "ED_FAIL", f"int_sep parse error: {exc}")
                            continue

                        sc_size = "N/A"
                        # ED / cluster generation
                        try:
                            clusters = generate_clusters(L, Nc, int_sep, v_sep_ratio)
                            sc_size = clusters.shape[1]
                            kpts = convert_site_clusters_to_k(clusters, L)
                            for sc_k in kpts:
                                k_list = [(float(k),) if np.ndim(k) == 0 else tuple(map(float, np.ravel(k))) for k in sc_k]
                                row(writer, Nc, sc_size, int_sep, U, V, t, "ED", k_list)
                        except Exception as exc:
                            row(writer, Nc, sc_size, int_sep, U, V, t, "ED_FAIL", str(exc))

                        # Finite DMRG
                        try:
                            _E, _psi, _fill = get_gnd(L=L, chi=CHI, U=U, t=t, mu=mu_0, V=V, V_sep=v_sep_ratio)
                            row(writer, Nc, sc_size, int_sep, U, V, t, "DMRG", "N/A")
                        except Exception as exc:
                            row(writer, Nc, sc_size, int_sep, U, V, t, "DMRG_FAIL", str(exc))

                        # iDMRG
                        try:
                            _E, _fill, _psi = run_dmrg_method(U=U, mu_0=mu_0, V=V, V_sep=v_sep_ratio, t=t, system_size=L, chi=CHI)
                            row(writer, Nc, sc_size, int_sep, U, V, t, "iDMRG", "N/A")
                        except Exception as exc:
                            row(writer, Nc, sc_size, int_sep, U, V, t, "iDMRG_FAIL", str(exc))
PY

echo "CSV written to $OUT_CSV"
