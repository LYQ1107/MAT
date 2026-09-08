from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from typing import Any

from mat import __version__
from mat.assets import AssetCatalog, AssetDownloader, AuthorizedProxyPolicy, DirectOnlyPolicy
from mat.backends.sleap_nn import SleapNNBackend
from mat.assets.catalog import AssetSpec
from mat.data.adapters import RatIDAdapter, PigReIDAdapter, PigTrackingAdapter, MultiCamCowsAdapter, SleapGerbilsAdapter
from mat.data.sanitize import validate_neutral_manifest
from mat.data.splits import freeze_by_field
from mat.core.validation import validate_protocol
from mat.core.errors import MATError, MissingAssetError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "configs/assets/catalog.yaml"


def _work_root(value: str | None = None) -> Path:
    return Path(value or os.environ.get("MAT_WORK_ROOT", "/data2/usr_for_deadline/MAT_workspace")).expanduser()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _proxy_names() -> dict[str, str]:
    return {name: ("SET" if name in os.environ else "UNSET")
            for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "no_proxy")}


def doctor(args) -> int:
    work = _work_root(args.work_root); work.mkdir(parents=True, exist_ok=True)
    route_lines = []
    try:
        route_lines = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=5).stdout.splitlines()
    except Exception as exc:
        route_lines = [f"unavailable:{type(exc).__name__}"]
    gpus = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        except Exception as exc:
            gpus = [f"unavailable:{type(exc).__name__}"]
    record = {
        "schema_version": "mat.doctor.v1", "status": "SMOKE_PASSED", "mat_version": __version__,
        "user": os.environ.get("USER", "unknown"), "host": platform.node(), "python": sys.version,
        "cwd": str(Path.cwd()), "repo": str(ROOT), "work_root": str(work),
        "proxy_variables": _proxy_names(), "application_proxy_bypass": True,
        "route_evidence": {"interfaces_command": "ip -brief addr (M0 shell audit)", "routes": route_lines,
                            "tun_visible": False, "transparent_proxy_excluded": False},
        "route_status": "DIRECT_ROUTE_UNVERIFIED", "bulk_asset_status": "BLOCKED_DIRECT_ROUTE",
        "disk": {"data1": shutil.disk_usage("/data1")._asdict(), "data2": shutil.disk_usage("/data2")._asdict()},
        "gpu_snapshot": gpus, "large_task_started": False,
        "notes": ["Only application-layer proxy bypass is established; route cannot be certified from user process.",
                   "No training or bulk transfer started; existing processes were not touched."],
    }
    _write_json(work / "doctor.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    return 0


def load_catalog(path: str | None) -> AssetCatalog:
    return AssetCatalog.from_yaml(Path(path) if path else DEFAULT_CATALOG)


def _network_policy(mode: str):
    """Build the explicitly selected network policy without exposing secrets."""
    if mode == "authorized-proxy":
        # Host scope comes from each catalog AssetSpec and is enforced by the
        # downloader.  Keeping this policy unbound avoids a hidden global host
        # exception when the user explicitly authorizes a different official
        # provider (for example Hugging Face model assets).
        return AuthorizedProxyPolicy()
    if mode == "direct-only":
        return DirectOnlyPolicy()
    raise ValueError(f"unknown network mode: {mode}")


def assets_plan(args) -> int:
    catalog = load_catalog(args.catalog)
    specs = catalog.for_phase(args.phase)
    selected_ids = set(getattr(args, "asset_id", None) or ())
    if selected_ids:
        unknown = selected_ids - set(catalog.assets)
        if unknown:
            raise MATError(f"unknown asset ids: {', '.join(sorted(unknown))}")
        specs = [a for a in specs if a.asset_id in selected_ids]
    specs = [asdict(a) for a in specs]
    work = _work_root(getattr(args, "work_root", None)); out = Path(args.output) if args.output else work / "assets" / f"asset_plan_{args.phase}.json"
    mode = getattr(args, "network_mode", "direct-only")
    policy = _network_policy(mode)
    plan = {"schema_version": "mat.asset_plan.v1", "phase": args.phase,
            "network_mode": mode, "route_status": policy.route_status,
            "bulk_status": "AUTHORIZED_PROXY" if mode == "authorized-proxy" else "BLOCKED_DIRECT_ROUTE",
            "assets": specs}
    _write_json(out, plan); print(out); return 0


def _materialize_sleap_asset(work: Path, receipt: dict[str, Any], asset_id: str) -> None:
    """Expose verified SLEAP objects at the stable dataset paths without a copy."""
    if not asset_id.startswith("sleap_gerbils_") or receipt.get("status") != "VERIFIED":
        return
    source = Path(receipt["path"])
    names = {
        "sleap_gerbils_train": "train.pkg.slp",
        "sleap_gerbils_val": "val.pkg.slp",
        "sleap_gerbils_test": "test.pkg.slp",
        "sleap_gerbils_clip": "example_5min.mp4",
        "sleap_gerbils_tracking": "example_tracking.slp",
    }
    name = names.get(asset_id)
    if not name:
        return
    target_dir = work / "datasets" / "sleap_gerbils"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    if target.exists():
        expected_size = int(receipt.get("downloaded_bytes", -1))
        expected_sha = receipt.get("sha256")
        if target.is_file() and target.stat().st_size == expected_size:
            if expected_sha:
                digest = hashlib.sha256()
                with target.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha:
                    raise MATError(f"refusing to reuse SLEAP asset with checksum mismatch: {target}")
            return
        raise MATError(f"refusing to replace existing SLEAP asset: {target}")
    try:
        os.link(source, target)
    except OSError:
        # A cross-filesystem destination is unusual here; retain atomicity if a
        # hard link is unavailable, while still avoiding partial publication.
        part = target.with_name(target.name + ".part")
        import shutil as _shutil
        _shutil.copyfile(source, part)
        os.replace(part, target)


def assets_fetch(args) -> int:
    raw = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    work = _work_root(args.work_root)
    downloader = AssetDownloader(work / "assets")
    mode = getattr(args, "network_mode", "direct-only")
    policy = _network_policy(mode)
    selected_ids = set(getattr(args, "asset_id", None) or ())
    receipts = []
    for item in raw.get("assets", []):
        if selected_ids and item.get("asset_id") not in selected_ids:
            continue
        asset = AssetSpec(**item)
        receipt = downloader.fetch(asset, policy)
        receipt_dict = asdict(receipt)
        receipts.append(receipt_dict)
        _materialize_sleap_asset(work, receipt_dict, asset.asset_id)
    if not receipts:
        raise MATError("asset fetch selected no assets")
    out = work / "assets" / "fetch_receipts.json"
    _write_json(out, {"schema_version": "mat.receipts.v1", "network_mode": mode, "receipts": receipts}); print(out)
    return 0 if all(r.get("status") == "VERIFIED" for r in receipts) else 2


def assets_import(args) -> int:
    catalog = load_catalog(args.catalog); asset = catalog.get(args.asset)
    receipt = AssetDownloader(_work_root(args.work_root) / "assets").import_local(asset, Path(args.source))
    out = _work_root(args.work_root) / "assets" / "receipts" / f"{args.asset}.json"; _write_json(out, asdict(receipt)); print(json.dumps(asdict(receipt), ensure_ascii=False, indent=2)); return 0 if receipt.status == "VERIFIED" else 2


ADAPTERS = {"rat_id": RatIDAdapter, "pig_reid": PigReIDAdapter, "pig_tracking": PigTrackingAdapter,
            "multicamcows": MultiCamCowsAdapter, "sleap_gerbils": SleapGerbilsAdapter}


def _adapter(name: str):
    if name not in ADAPTERS: raise ValueError(f"unknown dataset {name}")
    return ADAPTERS[name]()


def data_inspect(args) -> int:
    work = _work_root(args.work_root)
    raw = Path(args.raw_root) if args.raw_root else (
        work / "datasets" / "sleap_gerbils" if args.dataset == "sleap_gerbils"
        else work / "assets" / "raw" / args.dataset
    )
    adapter = _adapter(args.dataset)
    if not raw.is_dir():
        result = {"dataset": args.dataset, "status": "BLOCKED_MISSING_ASSET", "raw_root": str(raw), "reason": "authorized raw directory is absent"}
    else:
        result = asdict(adapter.inspect(raw)); result["status"] = "SMOKE_PASSED"
    out = (work / "prepared" / args.dataset / "manifests" / "inventory.json"
           if args.dataset == "sleap_gerbils" else work / "assets" / "manifests" / args.dataset / "inventory.json")
    _write_json(out, result); print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result.get("status") == "SMOKE_PASSED" else 2


def data_prepare(args) -> int:
    work = _work_root(args.work_root)
    raw = Path(args.raw_root) if args.raw_root else (
        work / "datasets" / "sleap_gerbils" if args.dataset == "sleap_gerbils"
        else work / "assets" / "raw" / args.dataset
    )
    if not raw.is_dir():
        print(json.dumps({"status": "BLOCKED_MISSING_ASSET", "raw_root": str(raw)}, ensure_ascii=False)); return 2
    output_root = work / "prepared" / args.dataset if args.dataset == "sleap_gerbils" else work / "assets" / "manifests" / args.dataset
    bundle = _adapter(args.dataset).build_manifests(raw, output_root)
    print(json.dumps({"status": "SMOKE_PASSED", "observations": str(bundle.observations), "truth": str(bundle.private_eval_truth)}, ensure_ascii=False, indent=2)); return 0


def _sleap_runtime_env(work: Path) -> dict[str, str]:
    """Build the explicit local SLEAP subprocess environment."""
    env = os.environ.copy()
    paths = [
        ROOT / "src",
        work / "env" / "skia_overlay",
        work / "env" / "sleap_site",
        work / "env" / "torch_overlay",
        work / "env" / "deps_overlay",
        Path("/home/lwr/anaconda3/envs/masaenv/lib/python3.11/site-packages"),
        Path("/home/lwr/anaconda3/envs/sam2long/lib/python3.11/site-packages"),
    ]
    old = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join(str(path) for path in paths if path.exists())
    env["PYTHONPATH"] = prefix + (os.pathsep + old if old else "")
    return env


def _git_state() -> dict[str, Any]:
    def run(command: list[str]) -> str | None:
        try:
            result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=5, check=True)
            return result.stdout.strip()
        except Exception:
            return None
    status = run(["git", "status", "--porcelain"])
    return {"git_commit": run(["git", "rev-parse", "HEAD"]), "git_dirty": bool(status)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_hashes(work: Path, required: dict[str, Path]) -> dict[str, str | None]:
    """Read verified fetch hashes when available; hash local files only as fallback."""
    by_name: dict[str, str] = {}
    receipts = work / "assets" / "fetch_receipts.json"
    if receipts.is_file():
        try:
            raw = json.loads(receipts.read_text(encoding="utf-8"))
            for receipt in raw.get("receipts", []):
                name = Path(str(receipt.get("path", ""))).name
                if receipt.get("status") == "VERIFIED" and receipt.get("sha256"):
                    by_name[name] = str(receipt["sha256"])
        except (OSError, ValueError, TypeError):
            pass
    result: dict[str, str | None] = {}
    for label, path in required.items():
        result[label] = by_name.get(path.name)
        if result[label] is None and path.is_file():
            result[label] = _sha256_file(path)
    return result


def _training_stats(work: Path, *, scope: str = "smoke") -> dict[str, Any]:
    root_name = "sleap_gerbils_pose_full" if scope == "full" else "sleap_gerbils_pose_smoke"
    logs = sorted((work / "runs" / root_name).rglob("training_log.csv"))
    if not logs:
        return {"actual_epochs": None, "optimizer_steps": None, "global_step": None,
                "training_log": None, "training_scope": scope}
    path = logs[-1]
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        epochs = len(rows)
        # SLEAP's CSV records one optimizer step per row for this smoke config;
        # retain the explicit checkpoint step below when it is available.
        steps = None
        checkpoints = sorted((work / "runs" / root_name).rglob("*.ckpt"))
        if checkpoints:
            metadata = _checkpoint_training_metadata(work, checkpoints[-1])
            steps = metadata.get("global_step")
        if steps is None and rows and scope == "smoke":
            # The configured smoke run intentionally uses one optimizer step
            # per epoch; this fallback remains tied to the observed CSV rows,
            # rather than inventing a training count for an absent log.
            steps = epochs
        return {"actual_epochs": epochs, "optimizer_steps": steps, "global_step": steps,
                "training_log": str(path), "training_scope": scope}
    except (OSError, csv.Error):
        return {"actual_epochs": None, "optimizer_steps": None, "global_step": None,
                "training_log": str(path), "training_scope": scope}


def _checkpoint_training_metadata(work: Path, checkpoint: Path) -> dict[str, Any]:
    """Read Lightning epoch/global_step through the isolated SLEAP runtime."""
    executable = work / "env" / "sleap_site" / "bin" / "sleap-nn"
    if not executable.is_file() or not checkpoint.is_file():
        return {}
    script = (
        "import json,sys,torch; x=torch.load(sys.argv[1],map_location='cpu'); "
        "print(json.dumps({'epoch':x.get('epoch'),'global_step':x.get('global_step')}))"
    )
    try:
        result = subprocess.run([_executable_python(str(executable)), "-c", script, str(checkpoint)],
                                env=_sleap_runtime_env(work), capture_output=True, text=True,
                                timeout=180, check=False)
        if result.returncode == 0:
            return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    return {}


def _executable_python(executable: str) -> str:
    """Use the interpreter named by a SLEAP launcher shebang when available."""
    path = Path(executable)
    try:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if first.startswith("#!"):
            candidate = first[2:].strip().split()[0]
            if Path(candidate).is_file():
                return candidate
    except (OSError, IndexError):
        pass
    return sys.executable


def _identity_runtime_python(work: Path) -> str | None:
    """Find a local interpreter that can load the offline torch+timm stack.

    The audit/control interpreter intentionally stays lightweight and has no
    PyTorch.  Identity inference is therefore dispatched to an already
    installed local runtime (never installed or downloaded implicitly).  The
    probe is import-only and does not touch the network.
    """
    candidates = [
        work / "env" / "identity_site" / "bin" / "python",
        Path("/home/lwr/anaconda3/envs/BoT-SORT/bin/python"),
        Path("/home/lwr/anaconda3/envs/Deepseek/bin/python"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            probe = subprocess.run(
                [str(candidate), "-c", "import torch, timm"],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except OSError:
            continue
        if probe.returncode == 0:
            return str(candidate)
    return None


def _dispatch_identity_runtime(args, work: Path) -> int | None:
    """Run the identity command in a local torch+timm runtime when needed.

    Returns the child status when dispatch occurred, otherwise ``None``.  An
    environment marker prevents recursion if the selected runtime is also
    missing a dependency; in that case the normal, explicit blocker is
    emitted by ``GlobalIdentityBackend.from_local``.
    """
    if os.environ.get("MAT_IDENTITY_RUNTIME_DISPATCHED") == "1":
        return None
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
        return None
    except Exception:
        pass
    executable = _identity_runtime_python(work)
    if executable is None:
        return None
    command = [executable, "-m", "mat.cli", "experiment", str(args.experiment)]
    if getattr(args, "config", None):
        command.extend(["--config", str(args.config)])
    if getattr(args, "identity_checkpoint", None):
        command.extend(["--identity-checkpoint", str(args.identity_checkpoint)])
    command.extend(["--work-root", str(work)])
    env = dict(os.environ)
    env["MAT_IDENTITY_RUNTIME_DISPATCHED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["WANDB_MODE"] = "offline"
    source_root = str(ROOT / "src")
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
    return int(result.returncode)


def _checkpoint_under(path: Path, root: Path) -> bool:
    """Return true only when ``path`` is physically inside ``root``."""
    try:
        path.resolve().relative_to(root.expanduser().resolve())
    except ValueError:
        return False
    return True


def resolve_smoke_checkpoint(work: Path, explicit: str | Path | None = None) -> Path | None:
    """Resolve a smoke checkpoint without consulting the formal-run tree."""
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        return candidate if candidate.is_file() else None
    root = work.expanduser().resolve() / "runs" / "sleap_gerbils_pose_smoke"
    candidates = sorted(path for path in root.rglob("best.ckpt") if path.is_file()) if root.is_dir() else []
    return candidates[0] if candidates else None


def resolve_full_checkpoint(work: Path, explicit: str | Path | None = None) -> Path | None:
    """Resolve only a ``best.ckpt`` below the dedicated full-run directory.

    In particular, this function deliberately returns ``None`` for a smoke
    checkpoint or for an arbitrary checkpoint outside the full tree.  A caller
    can therefore surface ``BLOCKED_MISSING_FULL_POSE_CHECKPOINT`` instead of
    silently evaluating a two-step smoke model.
    """
    root = work.expanduser().resolve() / "runs" / "sleap_gerbils_pose_full"
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if (candidate.is_file() and candidate.name.lower() == "best.ckpt"
                and _checkpoint_under(candidate, root)):
            return candidate
        return None
    candidates = sorted(path for path in root.rglob("best.ckpt") if path.is_file()) if root.is_dir() else []
    return candidates[0] if candidates else None


def _gpu_snapshot() -> list[dict[str, Any]]:
    """Read a compact GPU snapshot without changing or terminating processes."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            rows.append({"index": int(fields[0]), "name": fields[1],
                         "memory_total_mib": int(float(fields[2])),
                         "memory_used_mib": int(float(fields[3])),
                         "utilization_gpu_percent": float(fields[4])})
        except ValueError:
            continue
    return rows


def _gpu_busy(rows: list[dict[str, Any]], gpu_index: int | None = None) -> bool:
    """Conservatively classify a GPU as busy from the read-only snapshot."""
    selected = [row for row in rows if gpu_index is None or row.get("index") == gpu_index]
    if not selected:
        return True
    # A few MiB of display/runtime reservation is harmless; sustained memory
    # or utilization above these small thresholds means another job owns it.
    return all(float(row.get("memory_used_mib", 0)) > 256
               or float(row.get("utilization_gpu_percent", 0)) > 5
               for row in selected)


def _choose_free_gpu(rows: list[dict[str, Any]]) -> int | None:
    """Choose the lowest-index GPU with no material competing allocation."""
    for row in sorted(rows, key=lambda value: int(value.get("index", 10**9))):
        if not _gpu_busy([row], int(row["index"])):
            return int(row["index"])
    return None


def _count_slp_instances(path: Path, backend: SleapNNBackend, env: dict[str, str]) -> dict[str, Any]:
    """Count public SLEAP prediction instances in a child runtime."""
    script = (
        "import json,sys; from sleap_io import load_slp; "
        "p=sys.argv[1]; labels=load_slp(p, open_videos=False, lazy=True); "
        "print(json.dumps({'labeled_frames':len(labels.labeled_frames),"
        "'instances':sum(len(f.instances) for f in labels.labeled_frames)})); labels.close()"
    )
    try:
        result = subprocess.run([_executable_python(backend.executable), "-c", script, str(path)],
                                env=env, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0:
            return {"status": "COUNT_FAILED", "error": result.stderr[-1000:]}
        return {"status": "COUNTED", **json.loads(result.stdout.strip().splitlines()[-1])}
    except Exception as exc:
        return {"status": "COUNT_FAILED", "error": f"{type(exc).__name__}: {exc}"}


def _write_full_training_plan(full_root: Path, config_path: Path, train_slp: Path,
                              val_slp: Path, max_epochs: int) -> None:
    """Persist the exact formal command without launching it."""
    command = [
        "sleap-nn", "train", str(config_path),
        f"data_config.train_labels_path=[{train_slp}]",
        f"data_config.val_labels_path=[{val_slp}]",
        f"trainer_config.max_epochs={int(max_epochs)}",
        "trainer_config.save_ckpt=true",
        f"trainer_config.ckpt_dir={full_root / 'models'}",
        "trainer_config.use_wandb=false",
    ]
    _write_json(full_root / "planned_command.json", {
        "schema_version": "mat.sleap_nn.full_plan.v1",
        "command": command,
        "max_epochs": int(max_epochs),
        "train_steps_per_epoch": None,
        "smoke_checkpoint_reference": None,
    })


def baseline_sleap_gerbils(args) -> int:
    """Run the auditable SLEAP pose chain with isolated smoke/full trees."""
    work = _work_root(args.work_root)
    # Keep the callable usable from older programmatic callers that supplied
    # only the pre-v2 Namespace fields; the CLI parser still provides all of
    # these explicitly.
    device = getattr(args, "device", "auto")
    gpu_index = getattr(args, "gpu_index", None)
    smoke_epochs = int(getattr(args, "smoke_epochs", 2))
    full_epochs = int(getattr(args, "full_epochs", 50))
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    checkpoint_arg = getattr(args, "checkpoint", None)
    executable_arg = getattr(args, "executable", None)
    clip_frames = getattr(args, "clip_frames", "0-2559")
    raw = work / "datasets" / "sleap_gerbils"
    run_dir = Path(args.run_dir) if args.run_dir else work / "runs" / "sleap_gerbils_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_stage = args.stage
    stage_aliases = {"smoke": "train-smoke", "predict": "test-full", "eval": "test-full",
                     "clip": "clip-full", "all": "all-full"}
    stage = stage_aliases.get(requested_stage, requested_stage)
    started_at = datetime.now(timezone.utc).isoformat()
    full_root = work / "runs" / "sleap_gerbils_pose_full"
    smoke_root = work / "runs" / "sleap_gerbils_pose_smoke"
    manifest: dict[str, Any] = {
        "schema_version": "mat.sleap_gerbils.run.v2", "status": "RUNNING",
        "stage": stage, "requested_stage": requested_stage, "run_dir": str(run_dir),
        "command": "mat baseline sleap-gerbils", "argv": [str(value) for value in sys.argv],
        "started_at": started_at, "start_time": started_at,
        "hostname": platform.node(), "device": device, "gpu_index": gpu_index,
        "seed": args.seed,
        "GPU": {"requested": device, "gpu_index": gpu_index,
                "visible_devices": str(gpu_index) if gpu_index is not None else None,
                "used": device != "cpu"},
        "clip_frames": clip_frames, **_git_state(), "dataset_root": str(raw),
        "blockers": [], "metrics": {}, "sleap_io_version": None,
        "sleap_nn_version": None, "torch_version": None,
        "identity_model_hash": None, "pose_checkpoint_hash": None,
        "gallery_start_version": None, "gallery_end_version": None,
        "gallery_versions": [], "smoke_checkpoint": None, "full_checkpoint": None,
    }
    if not raw.is_dir():
        manifest.update({"status": "BLOCKED_MISSING_ASSET", "blockers": ["missing official SLEAP gerbil directory"]})
        _write_json(run_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2)); return 2
    required = {"train": raw / "train.pkg.slp", "val": raw / "val.pkg.slp",
                "test": raw / "test.pkg.slp", "clip": raw / "example_5min.mp4",
                "tracking": raw / "example_tracking.slp"}
    missing = [f"{name}:{path}" for name, path in required.items() if not path.is_file()]
    if missing:
        manifest.update({"status": "BLOCKED_MISSING_ASSET", "blockers": missing})
        _write_json(run_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2)); return 2
    manifest["dataset_sha256"] = _asset_hashes(work, required)
    split_hash_input = {name: manifest["dataset_sha256"].get(name) for name in ("train", "val", "test")}
    manifest["split_sha256"] = hashlib.sha256(json.dumps(split_hash_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    if stage in {"inspect", "prepare", "all-full"}:
        manifest["inventory"] = asdict(SleapGerbilsAdapter().inspect(raw))
        if stage in {"prepare", "all-full"}:
            observations = work / "prepared" / "sleap_gerbils" / "manifests" / "observations.jsonl"
            if not observations.is_file():
                bundle = SleapGerbilsAdapter().build_manifests(raw, work / "prepared" / "sleap_gerbils")
                manifest["prepared"] = {"observations": str(bundle.observations), "truth": str(bundle.private_eval_truth)}
            else:
                manifest["prepared"] = {"observations": str(observations), "status": "EXISTING_NOT_REBUILT"}

    needs_runtime = stage in {"runtime", "train-smoke", "train-full", "test-full", "clip-full", "all-full"}
    backend = None
    env: dict[str, str] | None = None
    if needs_runtime:
        env = _sleap_runtime_env(work)
        # These changes are confined to the SLEAP child process.  The Codex
        # control process and its ambient proxy/GPU environment are untouched.
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        backend = SleapNNBackend(
            executable=executable_arg or str(work / "env" / "sleap_site" / "bin" / "sleap-nn"),
            device=device, env=env,
        )
        if stage in {"runtime", "all-full"}:
            manifest["runtime"] = backend.verify_runtime()
            manifest["sleap_nn_version"] = manifest["runtime"].get("version", {}).get("stdout_first_line")
            try:
                probe = subprocess.run([_executable_python(backend.executable), "-c", "import torch; print(torch.__version__)"],
                                       env=env, capture_output=True, text=True, timeout=30, check=False)
                manifest["torch_version"] = probe.stdout.strip() or None
            except Exception:
                manifest["torch_version"] = None
            try:
                probe = subprocess.run([_executable_python(backend.executable), "-c", "import sleap_io; print(getattr(sleap_io, '__version__', 'unknown'))"],
                                       env=env, capture_output=True, text=True, timeout=30, check=False)
                manifest["sleap_io_version"] = probe.stdout.strip() or None
            except Exception:
                manifest["sleap_io_version"] = None

    if stage == "train-smoke":
        assert backend is not None
        smoke_config_dir = smoke_root / "config"
        smoke_config = smoke_config_dir / "training_config.yaml"
        if not smoke_config.is_file():
            generated = backend.generate_config(required["train"], smoke_config_dir)
            smoke_config = next((p for p in generated if p.name == "training_config.yaml"), generated[0])
        checkpoint = resolve_smoke_checkpoint(work, checkpoint_arg)
        if checkpoint is None:
            checkpoint = backend.train(smoke_config, required["train"], required["val"], smoke_root,
                                       smoke_epochs, train_steps_per_epoch=1)
        manifest["smoke_checkpoint"] = str(checkpoint)
        manifest["pose_checkpoint"] = str(checkpoint)
        manifest["pose_checkpoint_sha256"] = _sha256_file(checkpoint)
        manifest["pose_checkpoint_hash"] = manifest["pose_checkpoint_sha256"]
        manifest["training_scope"] = "smoke"
        manifest["training_steps_override"] = 1

    if stage in {"train-full", "all-full"}:
        assert backend is not None and env is not None
        full_root.mkdir(parents=True, exist_ok=True)
        full_config = full_root / "training_config.yaml"
        if not full_config.is_file():
            generated = backend.generate_config(required["train"], full_root)
            generated_config = next((p for p in generated if p.name == "training_config.yaml"), generated[0])
            # Keep the generated file as the starting point, then enforce the
            # formal-run invariants below even when a previous interrupted run
            # left a config behind.
            if generated_config != full_config:
                full_config.write_text(generated_config.read_text(encoding="utf-8"), encoding="utf-8")
        import yaml
        raw_config = yaml.safe_load(full_config.read_text(encoding="utf-8")) or {}
        trainer = raw_config.setdefault("trainer_config", {})
        trainer["max_epochs"] = full_epochs
        trainer.pop("train_steps_per_epoch", None)
        trainer["resume_ckpt_path"] = None
        full_config.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
        _write_full_training_plan(full_root, full_config, required["train"], required["val"], full_epochs)
        manifest["full_training_plan"] = str(full_root / "planned_command.json")
        manifest["training_scope"] = "full"
        manifest["full_epochs_requested"] = full_epochs
        manifest["training_steps_override"] = None
        gpu_rows = _gpu_snapshot()
        manifest["gpu_snapshot_before_full"] = gpu_rows
        if device == "cpu":
            manifest["blockers"].append("formal full training is GPU-only; CPU 50-epoch execution is not authorized")
            manifest["status"] = "BLOCKED_GPU_BUSY"
        elif _gpu_busy(gpu_rows, gpu_index):
            manifest["blockers"].append("all requested GPUs are busy or unavailable; no competing process was touched")
            manifest["status"] = "BLOCKED_GPU_BUSY"
        else:
            selected_gpu = gpu_index if gpu_index is not None else _choose_free_gpu(gpu_rows)
            if selected_gpu is None:
                manifest["blockers"].append("all requested GPUs are busy or unavailable; no competing process was touched")
                manifest["status"] = "BLOCKED_GPU_BUSY"
                selected_gpu = None
            else:
                # Scope GPU visibility to the SLEAP child only, including the
                # automatically selected free device.
                env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
                backend.env = env
                manifest["gpu_index"] = selected_gpu
                manifest["GPU"]["gpu_index"] = selected_gpu
                manifest["GPU"]["visible_devices"] = str(selected_gpu)
            resume = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
        if manifest.get("status") != "BLOCKED_GPU_BUSY":
            checkpoint = backend.train(full_config, required["train"], required["val"], full_root,
                                       full_epochs, resume_checkpoint=resume)
            manifest["full_checkpoint"] = str(checkpoint)
            manifest["pose_checkpoint"] = str(checkpoint)
            manifest["pose_checkpoint_sha256"] = _sha256_file(checkpoint)
            manifest["pose_checkpoint_hash"] = manifest["pose_checkpoint_sha256"]

    if stage in {"test-full", "clip-full", "all-full"}:
        assert backend is not None and env is not None
        checkpoint = resolve_full_checkpoint(work, checkpoint_arg)
        if checkpoint is None:
            manifest["blockers"].append("BLOCKED_MISSING_FULL_POSE_CHECKPOINT")
        else:
            manifest["full_checkpoint"] = str(checkpoint)
            manifest["pose_checkpoint"] = str(checkpoint)
            checkpoint_sha = _sha256_file(checkpoint)
            manifest["pose_checkpoint_sha256"] = checkpoint_sha
            manifest["pose_checkpoint_hash"] = checkpoint_sha
            # Every formal prediction artifact is checkpoint-addressed and
            # lives below the full tree.  This prevents an older smoke
            # ``run_dir/test_predictions.slp`` (or a different full model)
            # from being silently reused because a path already exists.
            checkpoint_tag = checkpoint_sha[:16]
            if stage in {"test-full", "all-full"}:
                validation_dir = full_root / "validation"
                validation_dir.mkdir(parents=True, exist_ok=True)
                threshold_results = []
                selected_threshold = None
                for threshold in (0.10, 0.15, 0.20, 0.25):
                    val_prediction = validation_dir / f"predictions_{checkpoint_tag}_t{threshold:.2f}.slp"
                    if not val_prediction.is_file():
                        backend.predict(required["val"], checkpoint, val_prediction, only_labeled_frames=True,
                                        peak_threshold=threshold)
                    count = _count_slp_instances(val_prediction, backend, env)
                    threshold_results.append({"threshold": threshold, **count})
                    if count.get("status") == "COUNTED" and int(count.get("instances", 0)) > 0:
                        selected_threshold = threshold
                        break
                manifest["validation_prediction_thresholds"] = threshold_results
                if selected_threshold is None:
                    manifest["blockers"].append("formal_validation_has_no_predicted_instances")
                else:
                    test_dir = full_root / "test"
                    test_dir.mkdir(parents=True, exist_ok=True)
                    prediction = test_dir / f"test_predictions_{checkpoint_tag}.slp"
                    if not prediction.is_file():
                        backend.predict(required["test"], checkpoint, prediction, only_labeled_frames=True,
                                        peak_threshold=selected_threshold)
                    manifest["test_prediction"] = str(prediction)
                    manifest["test_prediction_count"] = _count_slp_instances(prediction, backend, env)
                    manifest["test_eval"] = backend.evaluate(required["test"], prediction,
                                                              test_dir / f"eval_{checkpoint_tag}")
                    if manifest["test_prediction_count"].get("instances", 0) == 0:
                        manifest["blockers"].append("formal_test_has_no_predicted_instances")
            if stage in {"clip-full", "all-full"}:
                clip_dir = full_root / "clip"
                clip_dir.mkdir(parents=True, exist_ok=True)
                clip_prediction = clip_dir / f"example_5min_predictions_{checkpoint_tag}.slp"
                if not clip_prediction.is_file():
                    backend.predict(required["clip"], checkpoint, clip_prediction, tracking=True, frames=clip_frames)
                manifest["clip_prediction"] = str(clip_prediction)
                manifest["clip_prediction_count"] = _count_slp_instances(clip_prediction, backend, env)

    # Never infer a full checkpoint from smoke.  Training statistics are kept
    # separately so the old two-step run cannot masquerade as formal training.
    if stage == "train-smoke":
        smoke_stats = _training_stats(work, scope="smoke")
        manifest.update(smoke_stats)
    elif stage in {"train-full", "all-full"}:
        manifest.update(_training_stats(work, scope="full"))
    elif stage in {"test-full", "clip-full"}:
        # A later evaluation invocation writes a fresh manifest in the same
        # run directory; retain the already measured formal training receipt
        # instead of dropping epoch/step provenance.
        manifest.update(_training_stats(work, scope="full"))
        plan = full_root / "planned_command.json"
        if plan.is_file():
            manifest["full_training_plan"] = str(plan)
    if manifest.get("status") == "RUNNING":
        manifest["status"] = "SUCCEEDED" if not manifest["blockers"] else "BLOCKED"
    manifest["blockers"] = sorted(set(manifest["blockers"]))
    if manifest["blockers"] and manifest["status"] == "SUCCEEDED":
        manifest["status"] = "BLOCKED"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["end_time"] = manifest["finished_at"]
    _write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0 if manifest["status"] == "SUCCEEDED" else 2


def experiment_identity(args) -> int:
    """Dispatch the file-backed B0 runner after a measured full pose run."""
    work = _work_root(args.work_root)
    default_configs = {
        "b0": ROOT / "configs" / "experiments" / "gerbils" / "B0_oracle_crop_diagnostic.yaml",
        "b1": ROOT / "configs" / "experiments" / "gerbils" / "B1_part_static.yaml",
        "b2": ROOT / "configs" / "experiments" / "gerbils" / "B2_learned_matcher.yaml",
        "o1": ROOT / "configs" / "experiments" / "gerbils" / "O1_safe_memory.yaml",
    }
    config = Path(args.config) if args.config else default_configs[args.experiment]
    run_dir = work / "runs" / f"{args.experiment}_gerbils"
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    checkpoint = (Path(args.identity_checkpoint).expanduser().resolve() if args.identity_checkpoint else
                  work / "assets" / "identity" / "megadescriptor_t_224" / "pytorch_model.bin")
    pose_checkpoint = resolve_full_checkpoint(work)
    result: dict[str, Any] = {"schema_version": "mat.identity_experiment.v1", "experiment": args.experiment, "config": str(config),
                              "command": "mat experiment " + args.experiment, "argv": [str(value) for value in sys.argv], "start_time": started_at,
                              "end_time": None, "hostname": platform.node(), "GPU": None,
                              "dataset_sha256": None, "split_sha256": None,
                              "sleap_io_version": None, "sleap_nn_version": None, "torch_version": None,
                              "identity_model_hash": None, "pose_checkpoint_hash": None,
                              "gallery_start_version": None, "gallery_end_version": None,
                              "status": "RUNNING", "optimizer_steps": None,
                              "checkpoint": str(checkpoint) if checkpoint else None,
                              "pose_checkpoint": str(pose_checkpoint) if pose_checkpoint else None,
                              "metrics": None, "blockers": []}
    if not config.is_file():
        result["status"] = "BLOCKED_MISSING_CONFIG"; result["blockers"].append(str(config))
    elif args.experiment != "b0":
        # B1/B2/O1 are intentionally not represented by a fake asset blocker;
        # they can only be promoted after a measured B0 result and currently
        # have no implemented file-backed runner.
        raise NotImplementedError(f"{args.experiment.upper()} file-backed runner is not implemented")
    elif pose_checkpoint is None:
        result["status"] = "BLOCKED_MISSING_FULL_POSE_CHECKPOINT"
        result["blockers"].append("full pose best.ckpt is required; smoke checkpoints are never accepted")
    elif not checkpoint.is_file():
        result["status"] = "BLOCKED_MISSING_IDENTITY_ASSET"
        result["blockers"].append("verified local MegaDescriptor-T-224 checkpoint/config is absent")
    else:
        baseline_manifest = work / "runs" / "sleap_gerbils_baseline" / "run_manifest.json"
        pose_manifest = json.loads(baseline_manifest.read_text(encoding="utf-8")) if baseline_manifest.is_file() else {}
        result["dataset_sha256"] = pose_manifest.get("dataset_sha256")
        result["split_sha256"] = pose_manifest.get("split_sha256")
        result["pose_checkpoint_hash"] = _sha256_file(pose_checkpoint)
        test_eval = pose_manifest.get("test_eval", {})
        test_count = pose_manifest.get("test_prediction_count", {}).get("instances", 0)
        if test_eval.get("status") != "SUCCEEDED" or int(test_count or 0) <= 0:
            result["status"] = "BLOCKED_POSE_TEST_NOT_SUCCEEDED"
            result["blockers"].append("full pose validation/test must have non-zero predictions and official metrics before B0")
            result["pose_test_status"] = test_eval.get("status")
        if result["status"] == "RUNNING":
            dispatched = _dispatch_identity_runtime(args, work)
            if dispatched is not None:
                # The local torch+timm runtime owns the child manifest and
                # result files; propagate its status without duplicating them
                # in the lightweight parent interpreter.
                return dispatched
        from mat.backends.wildlife import GlobalIdentityBackend
        model_config = checkpoint.with_name("config.json")
        if result["status"] == "RUNNING":
            try:
                encoder = GlobalIdentityBackend.from_local(checkpoint, config_path=model_config)
                from mat.experiments.gerbil_identity import (load_prepared_gerbil_samples,
                    load_private_gerbil_truth, load_session_inventory,
                    run_b0_oracle_crop_diagnostic)
                samples, _ = load_prepared_gerbil_samples(work)
                truth = load_private_gerbil_truth(work)
                sessions = load_session_inventory(work)
                gallery_path = run_dir / "gallery.sqlite"
                gallery = __import__("mat.identity.gallery", fromlist=["GalleryStore"]).GalleryStore(gallery_path)
                b0 = run_b0_oracle_crop_diagnostic(samples, truth, encoder, gallery, sessions,
                                                   cohort_uid="gerbils-b0-oracle")
                b0.write(run_dir / "b0_result.json")
                gallery.close()
                result["status"] = b0.status
                result["metrics"] = b0.metrics
                result["s0_sessions"] = list(b0.s0_sessions)
                result["anchor_counts"] = b0.anchor_counts
                result["gallery_start_version"] = b0.gallery_version
                result["gallery_end_version"] = b0.gallery_version
                if b0.blocker:
                    result["blockers"].append(b0.blocker)
            except Exception as exc:
                result["status"] = "BLOCKED_IDENTITY_RUNTIME"
                result["blockers"].append(f"{type(exc).__name__}: {exc}")
        result["identity_model_hash"] = _sha256_file(checkpoint)
    result["end_time"] = datetime.now(timezone.utc).isoformat()
    result["blockers"] = sorted(set(result["blockers"]))
    if result["status"] == "RUNNING":
        result["status"] = "SUCCEEDED" if not result["blockers"] else "BLOCKED"
    _write_json(run_dir / "run_manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result["status"] == "SUCCEEDED" else 2


def split_freeze(args) -> int:
    split = freeze_by_field(Path(args.manifest), args.field, args.seed)
    out = Path(args.output) if args.output else _work_root(args.work_root) / "assets" / "manifests" / "splits" / f"{split.split_id}.json"
    split.write(out); print(out); return 0


def blocked_stage(args) -> int:
    work = _work_root(getattr(args, "work_root", None)); stage = args.stage
    reason = {"features": "BLOCKED_MISSING_ASSET", "reference": "BLOCKED_MISSING_FRONTEND", "enroll": "BLOCKED_MISSING_FRONTEND", "run": "BLOCKED_MISSING_FRONTEND", "evaluate": "BLOCKED_MISSING_TRUTH"}.get(stage, "BLOCKED")
    out = work / "runs" / f"{stage}_status.json"; _write_json(out, {"status": reason, "stage": stage, "optimizer_steps": None, "checkpoint": None, "reason": "P1/P2 assets or verified upstream runtime are not available"}); print(json.dumps({"status": reason, "path": str(out)}, ensure_ascii=False)); return 2


def train_source(args) -> int:
    work = _work_root(args.work_root); out = work / "runs" / "source_training"; out.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "BLOCKED_MISSING_ASSET", "seed": args.seed, "optimizer_steps": None, "checkpoint": None, "reason": "no verified local MegaDescriptor/PyTorch training asset; no fabricated run"}
    _write_json(out / "training_receipt.json", receipt); print(json.dumps(receipt, ensure_ascii=False, indent=2)); return 2


def config_validate(args) -> int:
    raw = _load_config(Path(args.config))
    validate_protocol(raw.get("protocol", {})); print("SMOKE_PASSED"); return 0


def _load_config(path: Path) -> dict[str, Any]:
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    parent_ref = raw.pop("extends", None)
    if not parent_ref:
        return raw
    parent_path = (path.parent / parent_ref).with_suffix(path.suffix) if not str(parent_ref).endswith(('.yaml', '.yml')) else path.parent / parent_ref
    parent = _load_config(parent_path)
    return _deep_merge(parent, raw)


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mat", description="MAT auditable longitudinal identity pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--work-root"); p.set_defaults(func=doctor)
    p = sub.add_parser("config-validate"); p.add_argument("--config", required=True); p.set_defaults(func=config_validate)
    sources = sub.add_parser("sources").add_subparsers(dest="sources_cmd", required=True)
    p = sources.add_parser("audit"); p.add_argument("--config", required=True); p.set_defaults(func=lambda a: (print(Path(a.config).read_text(encoding="utf-8")), 0)[1])
    assets = sub.add_parser("assets").add_subparsers(dest="assets_cmd", required=True)
    p = assets.add_parser("plan"); p.add_argument("--catalog", required=True); p.add_argument("--phase", choices=["P0", "P1", "P2", "P3"], required=True); p.add_argument("--network-mode", choices=["direct-only", "authorized-proxy"], default="direct-only"); p.add_argument("--asset-id", action="append"); p.add_argument("--work-root"); p.add_argument("--output"); p.set_defaults(func=assets_plan)
    p = assets.add_parser("fetch"); p.add_argument("--plan", required=True); p.add_argument("--work-root"); p.add_argument("--network-mode", choices=["direct-only", "authorized-proxy"], default="direct-only"); p.add_argument("--asset-id", action="append"); p.set_defaults(func=assets_fetch)
    p = assets.add_parser("import"); p.add_argument("--asset", required=True); p.add_argument("--source", required=True); p.add_argument("--catalog"); p.add_argument("--work-root"); p.set_defaults(func=assets_import)
    data = sub.add_parser("data").add_subparsers(dest="data_cmd", required=True)
    p = data.add_parser("inspect"); p.add_argument("--dataset", choices=sorted(ADAPTERS), required=True); p.add_argument("--work-root"); p.add_argument("--raw-root"); p.set_defaults(func=data_inspect)
    p = data.add_parser("prepare"); p.add_argument("--dataset", choices=sorted(ADAPTERS), required=True); p.add_argument("--work-root"); p.add_argument("--raw-root"); p.set_defaults(func=data_prepare)
    baseline = sub.add_parser("baseline", help="run a real public-upstream baseline")
    bp = baseline.add_subparsers(dest="baseline_cmd", required=True)
    p = bp.add_parser("sleap-gerbils", help="SLEAP-NN pose baseline on the fixed NYU gerbil dataset")
    p.add_argument("--stage", choices=["inspect", "prepare", "runtime", "train-smoke", "train-full",
                                        "test-full", "clip-full", "all-full",
                                        # Compatibility aliases are mapped to the explicit scopes above.
                                        "smoke", "predict", "eval", "clip", "all"], default="all-full")
    p.add_argument("--work-root")
    p.add_argument("--run-dir")
    p.add_argument("--executable")
    p.add_argument("--checkpoint")
    p.add_argument("--device", choices=["cpu", "auto"], default="auto")
    p.add_argument("--gpu-index", type=int)
    p.add_argument("--smoke-epochs", type=int, default=2)
    p.add_argument("--full-epochs", type=int, default=50)
    p.add_argument("--max-epochs", type=int, help=argparse.SUPPRESS)
    p.add_argument("--resume-checkpoint")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--clip-frames", default="0-2559")
    p.set_defaults(func=baseline_sleap_gerbils)
    experiments = sub.add_parser("experiment", help="record an identity experiment attempt")
    ep = experiments.add_subparsers(dest="experiment", required=True)
    for name in ("b0", "b1", "b2", "o1"):
        p = ep.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--identity-checkpoint")
        p.add_argument("--work-root")
        p.set_defaults(func=experiment_identity)
    split = sub.add_parser("split").add_subparsers(dest="split_cmd", required=True)
    p = split.add_parser("freeze"); p.add_argument("--manifest", required=True); p.add_argument("--field", default="cohort_uid"); p.add_argument("--seed", type=int, default=17); p.add_argument("--output"); p.add_argument("--work-root"); p.set_defaults(func=split_freeze)
    p = sub.add_parser("features").add_subparsers(dest="features_cmd", required=True).add_parser("extract"); p.add_argument("--config", required=True); p.add_argument("--scope", choices=["pilot", "full"], default="pilot"); p.add_argument("--work-root"); p.set_defaults(func=lambda a: blocked_stage(type("Args", (), {"stage":"features", "work_root":a.work_root})()))
    p = sub.add_parser("train").add_subparsers(dest="train_cmd", required=True).add_parser("source"); p.add_argument("--config", required=True); p.add_argument("--scope", choices=["pilot", "full"], default="pilot"); p.add_argument("--seed", type=int, default=17); p.add_argument("--work-root"); p.set_defaults(func=train_source)
    for name, command in [("reference", "export"), ("enroll", None), ("run", "cohort"), ("evaluate", None)]:
        parent = sub.add_parser(name)
        if command: parent = parent.add_subparsers(dest=f"{name}_cmd", required=True).add_parser(command)
        parent.add_argument("--config", required=False); parent.add_argument("--work-root")
        if name == "enroll": parent.add_argument("--mode", choices=["auto", "human", "oracle_reference"], required=True)
        if name == "evaluate": parent.add_argument("--run"); parent.add_argument("--truth")
        parent.set_defaults(func=lambda a, s=name: blocked_stage(type("Args", (), {"stage":s, "work_root":a.work_root})()))
    return parser


def main(argv=None) -> int:
    parser = build_parser(); args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MATError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
