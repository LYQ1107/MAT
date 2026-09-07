from __future__ import annotations

from pathlib import Path
import subprocess

from mat.core.errors import DependencyUnavailableError


class IdTrackerBaselineRunner:
    def __init__(self, executable: str = "idmatcherai", *, env: dict[str, str] | None = None):
        self.executable = executable
        self.env = env

    def run(self, master_session: Path, matching_sessions: list[Path], log_path: Path, timeout: int = 3600):
        if not master_session.is_dir() or any(not p.is_dir() for p in matching_sessions):
            raise ValueError("idtracker.ai requires real session directories")
        command = [self.executable, str(master_session), *(str(p) for p in matching_sessions)]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True,
                                       timeout=timeout, env=self.env)
        except FileNotFoundError as exc:
            raise DependencyUnavailableError(f"{self.executable} is not installed") from exc
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
        return completed.returncode

