"""SSH and rsync tools for communicating with the remote instance."""

import atexit
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _resolve_ssh_key(key_path: str) -> str:
    """Resolve SSH key path, falling back to RUNPOD_SSH_KEY env var.

    If the key file doesn't exist on disk, check for a RUNPOD_SSH_KEY
    environment variable containing the private key contents. Write it
    to a temp file with correct permissions and return that path.
    This enables running in cloud environments (e.g. claude.ai/code)
    where the key is stored as a secret rather than a file.
    """
    expanded = str(Path(key_path).expanduser())
    if os.path.exists(expanded):
        return expanded

    key_data = os.environ.get("RUNPOD_SSH_KEY")
    if not key_data:
        return expanded  # Let SSH fail with its own error message

    # Write key to a secure temp file
    fd = tempfile.NamedTemporaryFile(
        mode="w", prefix="runpod_key_", suffix="", delete=False,
    )
    fd.write(key_data)
    if not key_data.endswith("\n"):
        fd.write("\n")
    fd.close()
    os.chmod(fd.name, 0o600)

    # Clean up on process exit
    atexit.register(lambda p=fd.name: os.unlink(p) if os.path.exists(p) else None)

    return fd.name


class RemoteConnection:
    """Manages SSH/rsync connection to a persistent remote instance."""

    def __init__(self, host: str, port: int, user: str, key_path: str):
        self.host = host
        self.port = port
        self.user = user
        self.key_path = _resolve_ssh_key(key_path)

    @property
    def _ssh_base(self) -> list[str]:
        return [
            "ssh",
            "-p", str(self.port),
            "-i", self.key_path,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            f"{self.user}@{self.host}",
        ]

    @property
    def _ssh_uri(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def _rsync_ssh(self) -> str:
        return f"ssh -p {self.port} -i {self.key_path} -o StrictHostKeyChecking=accept-new"

    def exec(
        self,
        command: str,
        cwd: str = "~",
        timeout: int = 3600,
    ) -> dict:
        """Run a shell command on the remote via SSH."""
        full_cmd = self._ssh_base + [f"cd {cwd} && {command}"]
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
            }

    def sync_to(
        self,
        local_dir: str,
        remote_dir: str,
        excludes: list[str],
    ) -> dict:
        """Rsync local directory to remote."""
        local_path = str(Path(local_dir).expanduser())
        if not local_path.endswith("/"):
            local_path += "/"

        cmd = [
            "rsync", "-avz", "--progress",
            "-e", self._rsync_ssh,
        ]
        for exc in excludes:
            cmd.extend(["--exclude", exc])
        cmd.extend([local_path, f"{self._ssh_uri}:{remote_dir}"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def sync_from(
        self,
        remote_dir: str,
        local_dir: str,
        patterns: Optional[list[str]] = None,
    ) -> dict:
        """Rsync results from remote to local."""
        if patterns is None:
            patterns = ["*.pkl", "*.html", "*.csv", "*.svg", "*.pdf"]

        remote_path = f"{self._ssh_uri}:{remote_dir}"
        if not remote_path.endswith("/"):
            remote_path += "/"

        local_path = str(Path(local_dir).expanduser())
        Path(local_path).mkdir(parents=True, exist_ok=True)

        cmd = [
            "rsync", "-avz", "--progress",
            "-e", self._rsync_ssh,
        ]
        for pat in patterns:
            cmd.extend(["--include", pat])
        cmd.extend(["--include", "*/", "--exclude", "*"])
        cmd.extend([remote_path, local_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
