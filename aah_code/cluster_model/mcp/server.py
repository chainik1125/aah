"""
MCP server for the AAH/AAHK figure pipeline.

Provides tools to run computations on a remote instance,
sync results, and generate publication-quality plots locally.

Usage:
    python -m aah_code.cluster_model.mcp.server
"""

from pathlib import Path
from typing import Optional

import yaml
from mcp.server.fastmcp import FastMCP

from .tools.remote import RemoteConnection
from .tools import compute as compute_tools
from .tools import plots as plot_tools

# ── Load config ─────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

REMOTE = CONFIG["remote"]
PATHS = CONFIG["paths"]
RSYNC_EXCLUDES = CONFIG["rsync"]["excludes"]

conn = RemoteConnection(
    host=REMOTE["host"],
    port=REMOTE["port"],
    user=REMOTE["user"],
    key_path=REMOTE["key_path"],
)

# ── MCP server ──────────────────────────────────────────────────────
mcp = FastMCP(name="aah-figures")


@mcp.tool()
def remote_exec(command: str, cwd: str = "~", timeout: int = 3600) -> dict:
    """
    Run a shell command on the remote instance via SSH.

    Args:
        command: Shell command to execute.
        cwd: Working directory on the remote (default: home).
        timeout: Timeout in seconds (default: 3600).

    Returns:
        Dict with exit_code, stdout, stderr.
    """
    return conn.exec(command, cwd=cwd, timeout=timeout)


@mcp.tool()
def sync_to_remote(remote_dir: Optional[str] = None) -> dict:
    """
    Rsync the project code to the remote instance.
    Excludes .venv, .git, large_files, __pycache__, *.pyc, logs.

    Args:
        remote_dir: Remote destination (default: from config).

    Returns:
        Dict with exit_code, stdout, stderr.
    """
    return conn.sync_to(
        local_dir=PATHS["project_root"],
        remote_dir=remote_dir or PATHS["remote_project_root"],
        excludes=RSYNC_EXCLUDES,
    )


@mcp.tool()
def sync_from_remote(
    remote_dir: str,
    local_dir: str,
    patterns: Optional[list[str]] = None,
) -> dict:
    """
    Rsync results from the remote instance to local machine.

    Args:
        remote_dir: Remote source directory (e.g. "~/aah/large_files/plots").
        local_dir: Local destination directory.
        patterns: File patterns to include (default: *.pkl, *.html, *.csv, *.svg, *.pdf).

    Returns:
        Dict with exit_code, stdout, stderr.
    """
    return conn.sync_from(
        remote_dir=remote_dir,
        local_dir=local_dir,
        patterns=patterns,
    )


@mcp.tool()
def run_computation(
    command: str,
    args: Optional[dict] = None,
    yaml_config: Optional[str] = None,
    timeout: int = 3600,
) -> dict:
    """
    Run a cluster_runner.py command on the remote instance.
    Automatically activates the venv and cd's to the project root.

    Available commands:
      - filling_cluster_sizes
      - filling_with_int_cluster_sizes
      - fixed_supercluster
      - compressibility_with_int_cluster_sizes
      - compressibility_cluster_sizes
      - compressibility_fixed_supercluster

    Args:
        command: cluster_runner command name.
        args: CLI arguments as key-value pairs (e.g. {"L": 48, "chi": 32}).
        yaml_config: YAML config file path (relative to project root).
        timeout: Timeout in seconds (default: 3600).

    Returns:
        Dict with exit_code, stdout, stderr.
    """
    return compute_tools.run_computation(
        conn=conn,
        remote_project_root=PATHS["remote_project_root"],
        command=command,
        args=args,
        yaml_config=yaml_config,
        timeout=timeout,
    )


@mcp.tool()
def setup_remote() -> dict:
    """
    Set up the remote environment: create venv and pip install the project.

    Returns:
        Dict with exit_code, stdout, stderr.
    """
    return compute_tools.setup_remote_env(conn, PATHS["remote_project_root"])


@mcp.tool()
def make_plots(figures: Optional[list[str]] = None) -> dict:
    """
    Run make_paper_plots.py locally to regenerate publication figures from pickle data.

    Args:
        figures: Which figures to generate. Options: "hub_comparison",
                 "v_convergence", "fixed_supercluster", or ["all"] (default).

    Returns:
        Dict with exit_code, stdout, stderr, generated_files.
    """
    return plot_tools.make_plots(
        project_root=PATHS["project_root"],
        figures=figures,
    )


@mcp.tool()
def replot_from_pickle(
    pickle_path: str,
    plot_relative_error: bool = True,
) -> dict:
    """
    Reload existing results from a pickle file and replot.
    Useful for generating raw energy plots from existing relative error data.

    Args:
        pickle_path: Path to the pickle file (relative or absolute).
        plot_relative_error: If True, plot relative error. If False, plot raw energies.

    Returns:
        Dict with exit_code, stdout, stderr.
    """
    return plot_tools.run_local_replot(
        project_root=PATHS["project_root"],
        pickle_path=pickle_path,
        plot_relative_error=plot_relative_error,
    )


@mcp.tool()
def run_computation_async(
    command: str,
    args: Optional[dict] = None,
    yaml_config: Optional[str] = None,
) -> dict:
    """
    Launch a computation in the background on the remote instance.
    Returns immediately with a PID for monitoring via check_progress.

    Use this for long-running computations that would exceed tool call timeouts.
    The computation writes progress to ~/aah/progress.json on the remote.

    Args:
        command: cluster_runner command name.
        args: CLI arguments as key-value pairs (e.g. {"L": 48, "n_jobs": 16}).
        yaml_config: YAML config file path (relative to project root).

    Returns:
        Dict with pid, log_file, and the command that was launched.
    """
    return compute_tools.run_computation_async(
        conn=conn,
        remote_project_root=PATHS["remote_project_root"],
        command=command,
        args=args,
        yaml_config=yaml_config,
    )


@mcp.tool()
def check_progress() -> dict:
    """
    Check the progress of a running computation on the remote instance.
    Reads progress.json and checks if the process is still alive.

    Returns:
        Dict with status ("running", "completed", "no_computation"),
        progress details (total_tasks, completed, ETA), and recent log lines.
    """
    return compute_tools.check_progress(
        conn=conn,
        remote_project_root=PATHS["remote_project_root"],
    )


if __name__ == "__main__":
    import os
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        port = int(os.environ.get("MCP_PORT", "8080"))
        mcp.run(transport="sse", host="0.0.0.0", port=port)
    else:
        mcp.run()
