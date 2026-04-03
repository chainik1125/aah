"""SSH and rsync tools for communicating with the remote instance."""

import subprocess
from pathlib import Path
from typing import Optional


class RemoteConnection:
    """Manages SSH/rsync connection to a persistent remote instance."""

    def __init__(self, host: str, port: int, user: str, key_path: str):
        self.host = host
        self.port = port
        self.user = user
        self.key_path = str(Path(key_path).expanduser())

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
