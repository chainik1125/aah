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
