"""High-level computation tools wrapping cluster_runner on the remote."""

from typing import Optional

from .remote import RemoteConnection


def setup_remote_env(conn: RemoteConnection, remote_project_root: str) -> dict:
    """Install the project in editable mode on the remote."""
    cmd = (
        f"cd {remote_project_root} && "
        "python3 -m venv .venv 2>/dev/null; "
        "source .venv/bin/activate && "
        "pip install -e . --quiet 2>&1 | tail -5"
    )
    return conn.exec(cmd, cwd=remote_project_root, timeout=300)


def run_computation(
    conn: RemoteConnection,
    remote_project_root: str,
    command: str,
    args: Optional[dict] = None,
    yaml_config: Optional[str] = None,
    timeout: int = 3600,
) -> dict:
    """
    Run a cluster_runner command on the remote.

    Args:
        conn: Remote connection.
        remote_project_root: Path to the project on the remote.
        command: cluster_runner command name (e.g. 'filling_with_int_cluster_sizes').
        args: CLI arguments as key-value pairs.
        yaml_config: Path to a YAML config file (relative to project root).
        timeout: Timeout in seconds.
    """
    parts = [
        f"cd {remote_project_root}",
        "source .venv/bin/activate",
        "python -m aah_code.cluster_model.cluster_runner",
    ]

    if yaml_config:
        parts[-1] += f" --config={yaml_config}"
    else:
        parts[-1] += f" {command}"
        if args:
            for key, value in args.items():
                if isinstance(value, bool):
                    if value:
                        parts[-1] += f" --{key}"
                    else:
                        parts[-1] += f" --no-{key}"
                elif isinstance(value, list):
                    parts[-1] += f" --{key}={','.join(str(v) for v in value)}"
                else:
                    parts[-1] += f" --{key}={value}"

    full_cmd = " && ".join(parts)
    return conn.exec(full_cmd, cwd=remote_project_root, timeout=timeout)


def _build_runner_cmd(
    remote_project_root: str,
    command: str,
    args: Optional[dict] = None,
    yaml_config: Optional[str] = None,
) -> str:
    """Build the cluster_runner command string."""
    cmd = "python -m aah_code.cluster_model.cluster_runner"
    if yaml_config:
        cmd += f" --config={yaml_config}"
    else:
        cmd += f" {command}"
        if args:
            for key, value in args.items():
                if isinstance(value, bool):
                    cmd += f" --{key}" if value else f" --no-{key}"
                elif isinstance(value, list):
                    cmd += f" --{key}={','.join(str(v) for v in value)}"
                else:
                    cmd += f" --{key}={value}"
    return cmd


def run_computation_async(
    conn: RemoteConnection,
    remote_project_root: str,
    command: str,
    args: Optional[dict] = None,
    yaml_config: Optional[str] = None,
) -> dict:
    """Launch a computation in the background via nohup, return immediately."""
    runner_cmd = _build_runner_cmd(remote_project_root, command, args, yaml_config)
    log_file = f"{remote_project_root}/computation.log"

    # Set env vars, activate venv, launch in background, capture PID
    full_cmd = (
        f"cd {remote_project_root} && "
        "source .venv/bin/activate && "
        "export CUDA_VISIBLE_DEVICES=-1 && "
        "export OMP_NUM_THREADS=1 && "
        "export OPENBLAS_NUM_THREADS=1 && "
        f"nohup {runner_cmd} > {log_file} 2>&1 & "
        "echo $!"
    )

    result = conn.exec(full_cmd, cwd=remote_project_root, timeout=30)
    pid = result["stdout"].strip().split("\n")[-1]

    return {
        "pid": pid,
        "log_file": log_file,
        "command": runner_cmd,
        "status": "launched" if result["exit_code"] == 0 else "error",
        "stderr": result["stderr"],
    }


def check_progress(
    conn: RemoteConnection,
    remote_project_root: str,
) -> dict:
    """Check progress of a running computation on the remote."""
    # Check for progress.json and running python processes
    cmd = (
        f"cd {remote_project_root} && "
        # Check if any cluster_runner process is alive
        "echo '=== PROCESS ===' && "
        "ps aux | grep cluster_runner | grep -v grep | head -3 && "
        # Read progress.json if it exists
        "echo '=== PROGRESS ===' && "
        "cat progress.json 2>/dev/null || echo 'no progress file' && "
        # Last 10 lines of computation log
        "echo '=== LOG ===' && "
        "tail -10 computation.log 2>/dev/null || echo 'no log file'"
    )

    result = conn.exec(cmd, cwd=remote_project_root, timeout=15)
    stdout = result["stdout"]

    # Parse sections
    sections = {}
    current = None
    for line in stdout.split("\n"):
        if line.startswith("=== ") and line.endswith(" ==="):
            current = line.strip("= ")
            sections[current] = []
        elif current:
            sections[current].append(line)

    process_lines = [l for l in sections.get("PROCESS", []) if l.strip()]
    progress_text = "\n".join(sections.get("PROGRESS", []))
    log_lines = sections.get("LOG", [])

    is_running = len(process_lines) > 0

    # Try to parse progress JSON
    progress = None
    try:
        import json
        progress = json.loads(progress_text)
    except Exception:
        pass

    return {
        "status": "running" if is_running else ("completed" if progress else "no_computation"),
        "progress": progress,
        "recent_log": "\n".join(log_lines[-10:]),
        "processes": process_lines,
    }
