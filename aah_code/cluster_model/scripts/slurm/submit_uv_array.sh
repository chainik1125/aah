#!/usr/bin/env bash
# Generate a manifest of (U,V) tasks and submit an array job on the cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# From .../aah/aah_code/cluster_model/scripts/slurm -> repo root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# ----------------------- User-configurable grids -----------------------
# Edit these lists to your desired sweep. Values are space-separated.
U_LIST=${U_LIST:-"0 5e-1 1 2 3 5 10"}
V_LIST=${V_LIST:-"1e-6 1 2 3 5 10"}
T_VALUE=${T_VALUE:-1.0}
L_VALUE=${L_VALUE:-120}
CHI_VALUE=${CHI_VALUE:-64}
# Target filling per site for fixed-filling workflows. Leave empty/"none" for default (grand-canonical).
SET_FILLING=${SET_FILLING:-""}
SOLVER_METHOD=${SOLVER_METHOD:-sparse_ED}
STATES_RETAINED=${STATES_RETAINED:-6}
V_SEP_RATIO=${V_SEP_RATIO:-"1,2"}
INT_SEP_RATIOS=${INT_SEP_RATIOS:-"{2:[1,2],4:[1,4],6:[1,6]}"}  # JSON-like dict or tuple
CLUSTER_SIZES=${CLUSTER_SIZES:-"2 4 6"}                  # space-separated list
INCLUDE_IDMRG=${INCLUDE_IDMRG:-true}
INCLUDE_FINITE_DMRG=${INCLUDE_FINITE_DMRG:-true}
INCLUDE_TIMING=${INCLUDE_TIMING:-false}
INCLUDE_TIMING_PLOT=${INCLUDE_TIMING_PLOT:-false}
JOB_NAME=${JOB_NAME:-sep_12}

VSEP_TAG=$(echo "$V_SEP_RATIO" | tr ',' '-')
RUN_ID=${RUN_ID:-"sep_${VSEP_TAG}_$(date +%Y%m%d_%H%M%S)"}

MANIFEST_PATH="${MANIFEST_PATH:-$PROJECT_ROOT/large_files/manifest_uv_${RUN_ID}.json}"
PARTIAL_DIR="${PARTIAL_DIR:-$PROJECT_ROOT/large_files/partials_uv/${RUN_ID}}"
RUN_SCRIPT="$SCRIPT_DIR/run_uv_array.sbatch"
export MANIFEST_PATH PARTIAL_DIR PROJECT_ROOT
# Export sweep parameters so the Python block sees the values defined above.
export U_LIST V_LIST T_VALUE L_VALUE CHI_VALUE SOLVER_METHOD STATES_RETAINED
export SET_FILLING
export V_SEP_RATIO INT_SEP_RATIOS CLUSTER_SIZES INCLUDE_IDMRG INCLUDE_FINITE_DMRG
export INCLUDE_TIMING INCLUDE_TIMING_PLOT JOB_NAME

# Merge job settings (array job uses run_uv_array.sbatch header)
MERGE_PARTITION=${MERGE_PARTITION:-IllinoisComputes}
MERGE_ACCOUNT=${MERGE_ACCOUNT:-bbradlyn-ic}
MERGE_TIME=${MERGE_TIME:-01:00:00}
MERGE_MEM=${MERGE_MEM:-8G}

mkdir -p "$(dirname "$MANIFEST_PATH")" "$PARTIAL_DIR" "$PROJECT_ROOT/logs"

# Build manifest
python3 - <<'PY'
import json, os, ast
from pathlib import Path

def get_env(name, default):
    return os.environ.get(name, default)

U_LIST = get_env("U_LIST", "0.5 1.0").split()
V_LIST = get_env("V_LIST", "0.1 1.0").split()
T_VALUE = float(get_env("T_VALUE", "1.0"))
L_VALUE = int(get_env("L_VALUE", "24"))
CHI_VALUE = int(get_env("CHI_VALUE", "32"))
SOLVER_METHOD = get_env("SOLVER_METHOD", "sparse_ED")
STATES_RETAINED = int(get_env("STATES_RETAINED", "6"))
V_SEP_RATIO = tuple(int(x) for x in get_env("V_SEP_RATIO", "1,2").split(","))
CLUSTER_SIZES = [int(x) for x in get_env("CLUSTER_SIZES", "2").split()]
INCLUDE_IDMRG = get_env("INCLUDE_IDMRG", "true").lower() == "true"
INCLUDE_FINITE_DMRG = get_env("INCLUDE_FINITE_DMRG", "true").lower() == "true"
INCLUDE_TIMING = get_env("INCLUDE_TIMING", "false").lower() == "true"
INCLUDE_TIMING_PLOT = get_env("INCLUDE_TIMING_PLOT", "false").lower() == "true"
SET_FILLING_RAW = get_env("SET_FILLING", "").strip()
if SET_FILLING_RAW.lower() in ("", "none", "null"):
    SET_FILLING = None
else:
    SET_FILLING = float(SET_FILLING_RAW)
DMRG_FIXED_FILLING = SET_FILLING is not None

int_sep_env = get_env("INT_SEP_RATIOS", "{2:[1,2]}")
try:
    parsed = ast.literal_eval(int_sep_env)
except Exception:
    parsed = int_sep_env

if isinstance(parsed, dict):
    norm = {}
    for k, v in parsed.items():
        try:
            nk = int(k)
        except Exception:
            nk = k
        norm[nk] = v
    int_sep_ratios = norm
else:
    int_sep_ratios = parsed

# If a dict is provided but some cluster_sizes are missing, fill them with the first ratio.
if isinstance(int_sep_ratios, dict):
    if len(int_sep_ratios) == 0:
        raise ValueError("INT_SEP_RATIOS dict is empty; provide at least one ratio.")
    fallback_ratio = next(iter(int_sep_ratios.values()))
    for Nc in CLUSTER_SIZES:
        if Nc not in int_sep_ratios:
            int_sep_ratios[Nc] = fallback_ratio

manifest_path = Path(get_env("MANIFEST_PATH", "manifest_uv.json"))
manifest = []
for U in U_LIST:
    for V in V_LIST:
        manifest.append({
            "U": float(U),
            "V": float(V),
            "t": T_VALUE,
            "L": L_VALUE,
            "chi": CHI_VALUE,
            "solver_method": SOLVER_METHOD,
            "states_retained": STATES_RETAINED,
            "v_sep_ratio": list(V_SEP_RATIO),
            # Broadcast allowed; compare_* will map single ratio to all cluster_sizes.
            "int_sep_ratios": int_sep_ratios,
            "cluster_sizes": CLUSTER_SIZES,
            "include_idmrg": INCLUDE_IDMRG,
            "include_finite_dmrg": INCLUDE_FINITE_DMRG,
            "include_timing": INCLUDE_TIMING,
            "include_timing_plot": INCLUDE_TIMING_PLOT,
            "set_filling": SET_FILLING,
            "dmrg_fixed_filling": DMRG_FIXED_FILLING,
        })

manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"Wrote manifest with {len(manifest)} tasks to {manifest_path}")
PY

TASK_COUNT=$(python3 - <<'PY'
import json, os
from pathlib import Path
mp = Path(os.environ["MANIFEST_PATH"])
data = json.loads(mp.read_text())
print(len(data))
PY
)

read MAX_NC VSEP_STR <<EOF
$(python3 - <<'PY'
import json, os
from pathlib import Path
mp = Path(os.environ["MANIFEST_PATH"])
data = json.loads(mp.read_text())
if not data:
    raise SystemExit("Manifest empty")
max_nc = max(max(entry.get("cluster_sizes", [0])) for entry in data)
v_sep = data[0].get("v_sep_ratio", [])
vsep_str = "-".join(str(x) for x in v_sep) if v_sep else "vsep"
print(max_nc)
print(vsep_str)
PY
)
EOF
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MERGE_OUTPUT="$PARTIAL_DIR/merged_v${VSEP_STR}_Nc${MAX_NC}_${TIMESTAMP}.pkl"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found on this host; run this script on the cluster login node." >&2
  exit 1
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi

if [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "No tasks to submit (manifest empty)." >&2
  exit 1
fi

echo "Submitting array with $TASK_COUNT tasks"
ARRAY_RANGE="0-$((TASK_COUNT-1))"

ARRAY_JOBID=$(sbatch --parsable --array=${ARRAY_RANGE} \
  --job-name="$JOB_NAME" \
  --export=ALL,MANIFEST_PATH="$MANIFEST_PATH",PARTIAL_DIR="$PARTIAL_DIR",PROJECT_ROOT="$PROJECT_ROOT" \
  "$RUN_SCRIPT")
echo "Array job submitted: $ARRAY_JOBID"

MERGE_CMD="cd '$PROJECT_ROOT' && source .venv/bin/activate && python aah/aah_code/cluster_model/merge_uv_results.py --partials '$PARTIAL_DIR/*.pkl' --output '$MERGE_OUTPUT'"

sbatch --dependency=afterok:${ARRAY_JOBID} \
  --job-name="${JOB_NAME}_merge" \
  --partition=$MERGE_PARTITION \
  --account=$MERGE_ACCOUNT \
  --time=$MERGE_TIME \
  --mem=$MERGE_MEM \
  --nodes=1 --ntasks=1 \
  --output=$PROJECT_ROOT/logs/merge_uv_%j.out \
  --error=$PROJECT_ROOT/logs/merge_uv_%j.err \
  --wrap="$MERGE_CMD"
