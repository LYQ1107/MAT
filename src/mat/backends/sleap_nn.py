"""Subprocess wrapper for the public SLEAP-NN command line interface.

The wrapper intentionally does not import SLEAP-NN internals.  Every operation
records the exact argv, captured stdout/stderr and return code before a
non-zero command is surfaced as a failure.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import os
import shlex
import subprocess
from typing import Any, Iterable

from mat.core.errors import MATError, MissingAssetError


class SleapNNCommandError(MATError):
    """A public SLEAP-NN command returned a non-zero status."""


class SleapNNBackend:
    def __init__(
        self,
        executable: str = "sleap-nn",
        device: str = "auto",
        env: dict[str, str] | None = None,
    ):
        self.executable = executable
        self.device = device
        # Copy the mapping so callers can safely reuse their environment.  The
        # mapping is never serialized into receipts (it may contain proxy URLs).
        self.env = dict(env) if env is not None else None

    @property
    def _work_root(self) -> Path:
        env = self.env if self.env is not None else os.environ
        return Path(env.get("MAT_WORK_ROOT", "/data2/usr_for_deadline/MAT_workspace"))

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _run(
        self,
        args: Iterable[str | Path],
        *,
        output_dir: Path,
        operation: str,
        cwd: Path | None = None,
        stdout_name: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = [self.executable, *(str(arg) for arg in args)]
        stdout_path = output_dir / (stdout_name or f"{operation}.stdout.txt")
        stderr_path = output_dir / f"{operation}.stderr.txt"
        receipt_path = output_dir / f"{operation}.command.json"
        started = self._utc_now()
        receipt: dict[str, Any] = {
            "schema_version": "mat.sleap_nn.command.v1",
            "operation": operation,
            "command": argv,
            "command_line": shlex.join(argv),
            "cwd": str(cwd.expanduser().resolve()) if cwd else str(Path.cwd()),
            "started_at": started,
            "pid": None,
            "return_code": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(cwd.expanduser().resolve()) if cwd else None,
                env=dict(self.env) if self.env is not None else None,
                # ``capture_output`` is a ``subprocess.run`` convenience
                # keyword and is not accepted by ``Popen``.  Use explicit
                # pipes so the audited stdout/stderr capture works on the
                # Python versions used by the isolated SLEAP runtime.
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
            receipt["pid"] = process.pid
            stdout, stderr = process.communicate()
            completed = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        except Exception as exc:
            # Persist the attempted command and failure type without including
            # the environment (which could contain proxy credentials).
            receipt.update({
                "finished_at": self._utc_now(),
                "return_code": None,
                "exception": f"{type(exc).__name__}: {exc}",
            })
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        receipt.update({"finished_at": self._utc_now(), "return_code": completed.returncode})
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if completed.returncode != 0:
            raise SleapNNCommandError(
                f"SLEAP-NN {operation} failed with return code {completed.returncode}; "
                f"see {stderr_path}"
            )
        return completed

    def verify_runtime(self) -> dict[str, Any]:
        """Run and save the version/help commands before any training command."""
        audit_dir = self._work_root / "upstream_audit" / "sleap_nn"
        commands = {
            "version": ["--version"],
            "config_help": ["config", "--help"],
            "train_help": ["train", "--help"],
            "predict_help": ["predict", "--help"],
            "eval_help": ["eval", "--help"],
        }
        result: dict[str, Any] = {"schema_version": "mat.sleap_nn.runtime.v1", "executable": self.executable}
        for name, args in commands.items():
            completed = self._run(args, output_dir=audit_dir, operation=f"audit_{name}", stdout_name=f"{name}.txt")
            # Keep a compact in-memory summary; complete output is in the audit
            # files and the command receipts.
            result[name] = {
                "return_code": completed.returncode,
                "stdout_path": str(audit_dir / f"{name}.txt"),
                "stderr_path": str(audit_dir / f"audit_{name}.stderr.txt"),
                "stdout_first_line": (completed.stdout or "").splitlines()[0] if completed.stdout else "",
            }
        (audit_dir / "runtime_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    def generate_config(
        self,
        train_slp: Path,
        output_dir: Path,
        pipeline: str = "bottomup",
    ) -> list[Path]:
        train_slp = train_slp.expanduser().resolve()
        if not train_slp.is_file():
            raise MissingAssetError(f"missing SLEAP train labels: {train_slp}")
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "training_config.yaml"
        self._run(
            ["config", train_slp, "--auto", "--pipeline", pipeline, "-o", config_path],
            output_dir=output_dir,
            operation="config",
        )
        generated = sorted(
            path for path in output_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
        if not generated:
            raise SleapNNCommandError(f"SLEAP-NN config completed without a YAML output in {output_dir}")
        return generated

    def train(
        self,
        config_path: Path,
        train_slp: Path,
        val_slp: Path,
        run_dir: Path,
        max_epochs: int,
        *,
        train_steps_per_epoch: int | None = None,
        resume_checkpoint: Path | None = None,
    ) -> Path:
        config_path = config_path.expanduser().resolve()
        train_slp = train_slp.expanduser().resolve()
        val_slp = val_slp.expanduser().resolve()
        run_dir = run_dir.expanduser().resolve()
        if not config_path.is_file() or not train_slp.is_file() or not val_slp.is_file():
            raise MissingAssetError("SLEAP-NN train requires existing config, train and validation labels")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if train_steps_per_epoch is not None and train_steps_per_epoch <= 0:
            raise ValueError("train_steps_per_epoch must be positive when supplied")
        if resume_checkpoint is not None:
            resume_checkpoint = resume_checkpoint.expanduser().resolve()
            if not resume_checkpoint.is_file():
                raise MissingAssetError(f"missing SLEAP resume checkpoint: {resume_checkpoint}")
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / "models"
        args = [
            "train", config_path,
            f"data_config.train_labels_path=[{train_slp}]",
            # SLEAP-NN's public training schema declares both train and
            # validation labels as lists. Passing a scalar path is rejected
            # by OmegaConf before training starts (``AnyNode is not a
            # ListConfig``), so keep the override shape identical to the
            # generated config.
            f"data_config.val_labels_path=[{val_slp}]",
            # ``validation_fraction`` is a typed float in SLEAP-NN 0.3.x;
            # when explicit validation labels are supplied it is ignored, so
            # leave the generated (0.1) value intact instead of assigning
            # ``null`` and failing schema validation.
            f"trainer_config.max_epochs={int(max_epochs)}",
            "trainer_config.save_ckpt=true",
            f"trainer_config.ckpt_dir={ckpt_dir}",
            "trainer_config.use_wandb=false",
        ]
        # A step override is intentionally explicit. It must never be inferred
        # from max_epochs, otherwise a formal run can silently become smoke.
        if train_steps_per_epoch is not None:
            args.extend([
                "trainer_config.min_train_steps_per_epoch=1",
                f"+trainer_config.train_steps_per_epoch={int(train_steps_per_epoch)}",
            ])
        if resume_checkpoint is not None:
            # This key is present in the audited public SLEAP-NN 0.3.3 schema.
            args.append(f"trainer_config.resume_ckpt_path={resume_checkpoint}")
        self._run(args, output_dir=run_dir, operation="train", cwd=run_dir)
        checkpoints = sorted(ckpt_dir.rglob("*.ckpt")) if ckpt_dir.exists() else sorted(run_dir.rglob("*.ckpt"))
        if not checkpoints:
            raise SleapNNCommandError(f"SLEAP-NN train returned success but produced no checkpoint in {run_dir}")
        preferred = [p for p in checkpoints if "best" in p.name.lower()]
        return preferred[0] if preferred else checkpoints[0]

    def predict(
        self,
        data_path: Path,
        model_path: Path,
        output_path: Path,
        tracking: bool = False,
        only_labeled_frames: bool = False,
        peak_threshold: float | None = None,
        max_instances: int | None = None,
        frames: str | None = None,
    ) -> Path:
        data_path = data_path.expanduser().resolve()
        model_path = model_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        if not data_path.exists():
            raise MissingAssetError(f"missing SLEAP prediction input: {data_path}")
        if not model_path.exists():
            raise MissingAssetError(f"missing SLEAP model/checkpoint: {model_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args: list[str | Path] = [
            "predict", "--data_path", data_path, "--model_paths", model_path,
            "--output_path", output_path, "--device", self.device,
        ]
        if tracking:
            args.append("--tracking")
        if only_labeled_frames:
            args.append("--only_labeled_frames")
        if peak_threshold is not None:
            if not 0.0 <= peak_threshold <= 1.0:
                raise ValueError("peak_threshold must be between 0 and 1")
            args.extend(["--peak_threshold", str(float(peak_threshold))])
        if max_instances is not None:
            if max_instances <= 0:
                raise ValueError("max_instances must be positive")
            args.extend(["--max_instances", str(int(max_instances))])
        if frames is not None:
            if not frames.strip():
                raise ValueError("frames must be a non-empty CLI frame range")
            args.extend(["--frames", frames])
        self._run(args, output_dir=output_path.parent, operation=f"predict_{output_path.stem}")
        if not output_path.is_file():
            raise SleapNNCommandError(f"SLEAP-NN predict completed without output: {output_path}")
        return output_path

    def evaluate(self, gt_slp: Path, pred_slp: Path, output_dir: Path) -> dict[str, Any]:
        gt_slp = gt_slp.expanduser().resolve()
        pred_slp = pred_slp.expanduser().resolve()
        output_dir = output_dir.expanduser().resolve()
        if not gt_slp.is_file() or not pred_slp.is_file():
            raise MissingAssetError("SLEAP-NN eval requires existing ground-truth and prediction labels")
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / "metrics_official.npz"
        self._run(
            ["eval", "-g", gt_slp, "-p", pred_slp, "-s", npz_path],
            output_dir=output_dir,
            operation="eval",
        )
        metrics: dict[str, Any] = {}
        if npz_path.is_file():
            import numpy as np
            with np.load(npz_path, allow_pickle=False) as values:
                for key in values.files:
                    value = values[key]
                    metrics[key] = value.item() if value.shape == () else value.tolist()
        result = {
            # The public evaluator intentionally emits no NPZ when there are
            # zero predicted instances.  Keep that outcome distinct from a
            # metric-bearing success so downstream reports cannot mistake an
            # empty prediction for a measured score.
            "status": "SUCCEEDED" if npz_path.is_file() else "SUCCEEDED_NO_PREDICTIONS",
            "metrics_npz": str(npz_path) if npz_path.is_file() else None,
            "metrics": metrics,
        }
        (output_dir / "metrics_official.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return result
