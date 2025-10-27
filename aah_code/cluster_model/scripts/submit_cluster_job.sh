#!/bin/bash
# Submit a cluster-model run to the Illinois Campus Cluster.
# The script syncs the repo to the remote project directory and fires off sbatch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk four levels up: scripts -> cluster_model -> aah_code -> aah -> repo root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
DEFAULT_SLURM_SCRIPT="aah/aah_code/cluster_model/scripts/slurm/run_convergence.sbatch"
DEFAULT_CONFIG_PATH="$PROJECT_ROOT/aah/pyproject.toml"

usage() {
    cat <<'EOF'
Usage: submit_cluster_job.sh [--config path] [SLURM_SCRIPT]

Options:
  --config PATH   Path to a TOML file providing remote and rsync settings.
                  Defaults to aah/pyproject.toml (if present).
  -h, --help      Show this message.

Arguments:
  SLURM_SCRIPT    Relative path (from repo root) to the sbatch file to submit.
                  Overrides any value provided via the config file.

Environment overrides:
  REMOTE_USER, REMOTE_HOST, REMOTE_PROJECT_ROOT
  SLURM_SCRIPT_RELATIVE, SYNC_EXTRA_EXCLUDES, RSYNC_DELETE
  CLUSTER_CONFIG_TOML, SBATCH_EXTRA_ARGS
  SSH_CONTROL_ENABLE, SSH_CONTROL_DIR, SSH_CONTROL_PATH, SSH_CONTROL_PERSIST

Any of these env vars take precedence over the config file values.
EOF
}

# -----------------------------------------------------------------------------
# Parse CLI flags
# -----------------------------------------------------------------------------

CONFIG_PATH="${CLUSTER_CONFIG_TOML:-$DEFAULT_CONFIG_PATH}"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
if ((${#POSITIONAL[@]})); then
    set -- "${POSITIONAL[@]}"
else
    set --
fi

CLI_SLURM_SCRIPT="${1:-}"
if [[ $# -gt 1 ]]; then
    echo "Unexpected arguments: $*" >&2
    usage
    exit 1
fi

# -----------------------------------------------------------------------------
# Load optional TOML configuration
# -----------------------------------------------------------------------------

config_exports=""
if [[ -n "$CONFIG_PATH" && -f "$CONFIG_PATH" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required to parse TOML configuration." >&2
        exit 1
    fi
    if ! config_exports="$(CONFIG_TOML_PATH="$CONFIG_PATH" python3 - <<'PY'
import ast
import os
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

path = Path(os.environ["CONFIG_TOML_PATH"])
if not path.exists():
    sys.exit(0)

cfg = load_toml(path)

def get_value(keys: str):
    cur = cfg
    for key in keys.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

mapping = {
    "tool.cluster.remote.user": "CONFIG_REMOTE_USER",
    "tool.cluster.remote.host": "CONFIG_REMOTE_HOST",
    "tool.cluster.remote.project_root": "CONFIG_REMOTE_PROJECT_ROOT",
    "tool.cluster.rsync.excludes": "CONFIG_SYNC_EXCLUDES",
    "tool.cluster.rsync.delete": "CONFIG_RSYNC_DELETE",
    "tool.cluster.sbatch.script": "CONFIG_SLURM_SCRIPT",
    "tool.cluster.sbatch.extra_args": "CONFIG_SLURM_EXTRA_ARGS",
}

for dotted, env_name in mapping.items():
    value = get_value(dotted)
    if value is None:
        continue
    if isinstance(value, list):
        value = ",".join(str(v) for v in value)
    elif isinstance(value, bool):
        value = "true" if value else "false"
    else:
        value = str(value)
    print(f"{env_name}={value!r}")
PY
)"; then
        echo "Failed to parse config at $CONFIG_PATH" >&2
        exit 1
    fi
    eval "$config_exports"
elif [[ -n "$CONFIG_PATH" && ! -f "$CONFIG_PATH" ]]; then
    echo "Warning: config file not found at $CONFIG_PATH (continuing with defaults)." >&2
fi

# -----------------------------------------------------------------------------
# Resolve final settings: env > CLI > config > defaults
# -----------------------------------------------------------------------------

REMOTE_USER="${REMOTE_USER:-${CONFIG_REMOTE_USER:-dmitry2}}"
REMOTE_HOST="${REMOTE_HOST:-${CONFIG_REMOTE_HOST:-cc-login.campuscluster.illinois.edu}}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-${CONFIG_REMOTE_PROJECT_ROOT:-/projects/illinois/eng/physics/bbradlyn/dmitry2/k_blocking/new_code}}"

if [[ -n "${SLURM_SCRIPT_RELATIVE:-}" ]]; then
    FINAL_SLURM_SCRIPT="$SLURM_SCRIPT_RELATIVE"
elif [[ -n "$CLI_SLURM_SCRIPT" ]]; then
    FINAL_SLURM_SCRIPT="$CLI_SLURM_SCRIPT"
elif [[ -n "${CONFIG_SLURM_SCRIPT:-}" ]]; then
    FINAL_SLURM_SCRIPT="$CONFIG_SLURM_SCRIPT"
else
    FINAL_SLURM_SCRIPT="$DEFAULT_SLURM_SCRIPT"
fi

SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-${CONFIG_SLURM_EXTRA_ARGS:-}}"

DEFAULT_EXCLUDES=(".git/" "large_files/")
RSYNC_EXCLUDES=("${DEFAULT_EXCLUDES[@]}")

append_excludes_from_csv() {
    local csv="$1"
    IFS=',' read -r -a tmp_arr <<< "$csv"
    for item in "${tmp_arr[@]}"; do
        local trimmed="${item#"${item%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
        [[ -n "$trimmed" ]] && RSYNC_EXCLUDES+=("$trimmed")
    done
}

if [[ -n "${CONFIG_SYNC_EXCLUDES:-}" ]]; then
    append_excludes_from_csv "$CONFIG_SYNC_EXCLUDES"
fi
if [[ -n "${SYNC_EXTRA_EXCLUDES:-}" ]]; then
    append_excludes_from_csv "$SYNC_EXTRA_EXCLUDES"
fi

RSYNC_DELETE_FLAG="--delete"
DELETE_PREF="${RSYNC_DELETE:-${CONFIG_RSYNC_DELETE:-true}}"
if [[ "$DELETE_PREF" =~ ^([Nn]o|[Ff]alse|0)$ ]]; then
    RSYNC_DELETE_FLAG=""
fi

control_opts=()
if [[ "${SSH_CONTROL_ENABLE:-1}" != "0" ]]; then
    SSH_CONTROL_DIR="${SSH_CONTROL_DIR:-$HOME/.ssh/cm}"
    mkdir -p "$SSH_CONTROL_DIR"
    SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-$SSH_CONTROL_DIR/%r@%h:%p}"
    SSH_CONTROL_PERSIST="${SSH_CONTROL_PERSIST:-yes}"
    control_opts=(-o "ControlMaster=auto" -o "ControlPath=$SSH_CONTROL_PATH" -o "ControlPersist=$SSH_CONTROL_PERSIST")
fi

LOCAL_SLURM_PATH="$PROJECT_ROOT/$FINAL_SLURM_SCRIPT"
if [[ ! -f "$LOCAL_SLURM_PATH" ]]; then
    echo "SLURM script not found at $LOCAL_SLURM_PATH" >&2
    echo "Pass the relative path as an argument or via the config." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Summary + confirmation
# -----------------------------------------------------------------------------

echo "Project root:     $PROJECT_ROOT"
echo "Remote directory: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_ROOT"
echo "SLURM script:     $FINAL_SLURM_SCRIPT"
if [[ -n "$SBATCH_EXTRA_ARGS" ]]; then
    echo "sbatch extra args: $SBATCH_EXTRA_ARGS"
fi
if [[ -n "$CONFIG_PATH" && -f "$CONFIG_PATH" ]]; then
    echo "Config file:       $CONFIG_PATH"
fi
echo "rsync excludes:   ${RSYNC_EXCLUDES[*]}"
[[ -z "$RSYNC_DELETE_FLAG" ]] && echo "rsync delete:      disabled"
if ((${#control_opts[@]})); then
    echo "SSH control path:  $SSH_CONTROL_PATH"
    echo "SSH control keep:  $SSH_CONTROL_PERSIST"
fi
echo
read -rp "Proceed with rsync + sbatch? [y/N] " reply
if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# -----------------------------------------------------------------------------
# Sync + submit
# -----------------------------------------------------------------------------

ssh_target="$REMOTE_USER@$REMOTE_HOST"

if ((${#control_opts[@]})); then
    if ! ssh "${control_opts[@]}" -O check "$ssh_target" >/dev/null 2>&1; then
        ssh "${control_opts[@]}" -fN "$ssh_target"
    fi
fi

rsync_cmd=(rsync -az)
[[ -n "$RSYNC_DELETE_FLAG" ]] && rsync_cmd+=("$RSYNC_DELETE_FLAG")
for pattern in "${RSYNC_EXCLUDES[@]}"; do
    rsync_cmd+=("--exclude=$pattern")
done
if ((${#control_opts[@]})); then
    rsync_cmd+=(-e "ssh ${control_opts[*]}")
fi
rsync_cmd+=("$PROJECT_ROOT/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PROJECT_ROOT/")

"${rsync_cmd[@]}"

REMOTE_SLURM_PATH="$REMOTE_PROJECT_ROOT/$FINAL_SLURM_SCRIPT"

ssh "${control_opts[@]}" "$ssh_target" bash <<EOF
set -euo pipefail
cd "$REMOTE_PROJECT_ROOT"
EXTRA_ARGS="$SBATCH_EXTRA_ARGS"
if [[ -n "\$EXTRA_ARGS" ]]; then
    sbatch \$EXTRA_ARGS "$REMOTE_SLURM_PATH"
else
    sbatch "$REMOTE_SLURM_PATH"
fi
EOF

echo "Submitted $FINAL_SLURM_SCRIPT via sbatch."
