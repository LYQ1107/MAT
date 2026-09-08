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
        # The current user authorization is scoped to the official SLEAP object
        # host.  This prevents accidentally sending legacy Rat/Pig/Cow assets
        # through the proxy in the same invocation.
        return AuthorizedProxyPolicy(allowed_hosts=frozenset({"storage.googleapis.com"}))
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


def _training_stats(work: Path) -> dict[str, Any]:
    logs = sorted((work / "runs" / "sleap_gerbils_pose_smoke").rglob("training_log.csv"))
    if not logs:
        return {"actual_epochs": None, "optimizer_steps": None, "training_log": None}
    path = logs[-1]
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        epochs = len(rows)
        # SLEAP's CSV records one optimizer step per row for this smoke config;
        # retain the explicit checkpoint step below when it is available.
        steps = None
        checkpoints = sorted(path.parent.parent.rglob("*.ckpt"))
        if checkpoints:
            try:
                import zipfile
                with zipfile.ZipFile(checkpoints[-1]) as archive:
                    text = archive.read("metadata.json").decode("utf-8", errors="replace")
                    metadata = json.loads(text)
                    steps = metadata.get("global_step")
            except Exception:
                pass
        if steps is None and rows:
            # The configured smoke run intentionally uses one optimizer step
            # per epoch; this fallback remains tied to the observed CSV rows,
            # rather than inventing a training count for an absent log.
            steps = epochs
        return {"actual_epochs": epochs, "optimizer_steps": steps, "training_log": str(path)}
    except (OSError, csv.Error):
        return {"actual_epochs": None, "optimizer_steps": None, "training_log": str(path)}


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


def baseline_sleap_gerbils(args) -> int:
    """Run real SLEAP baseline stages and persist a run manifest."""
    work = _work_root(args.work_root)
    raw = work / "datasets" / "sleap_gerbils"
    run_dir = Path(args.run_dir) if args.run_dir else work / "runs" / "sleap_gerbils_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": "mat.sleap_gerbils.run.v1", "status": "RUNNING",
        "stage": stage, "run_dir": str(run_dir),
        "command": "mat baseline sleap-gerbils", "argv": [str(value) for value in sys.argv],
        "started_at": started_at, "start_time": started_at,
        "hostname": platform.node(), "device": args.device, "seed": args.seed,
        "GPU": {"requested": args.device, "visible_devices": "" if args.device == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES"), "used": args.device != "cpu"},
        "clip_frames": args.clip_frames, **_git_state(), "dataset_root": str(raw),
        "blockers": [], "metrics": {}, "sleap_io_version": None,
        "sleap_nn_version": None, "torch_version": None,
        "identity_model_hash": None, "pose_checkpoint_hash": None,
        "gallery_start_version": None, "gallery_end_version": None,
        "gallery_versions": [],
    }
    if not raw.is_dir():
        manifest.update({"status": "BLOCKED_MISSING_ASSET", "blockers": ["missing official SLEAP gerbil directory"]})
        _write_json(run_dir / "run_manifest.json", manifest); print(json.dumps(manifest, ensure_ascii=False, indent=2)); return 2
    required = {
        "train": raw / "train.pkg.slp", "val": raw / "val.pkg.slp",
        "test": raw / "test.pkg.slp", "clip": raw / "example_5min.mp4",
        "tracking": raw / "example_tracking.slp",
    }
    missing = [f"{name}:{path}" for name, path in required.items() if not path.is_file()]
    if missing:
        manifest.update({"status": "BLOCKED_MISSING_ASSET", "blockers": missing})
        _write_json(run_dir / "run_manifest.json", manifest); print(json.dumps(manifest, ensure_ascii=False, indent=2)); return 2
    manifest["dataset_sha256"] = _asset_hashes(work, required)
    split_hash_input = {name: manifest["dataset_sha256"].get(name) for name in ("train", "val", "test")}
    manifest["split_sha256"] = hashlib.sha256(
        json.dumps(split_hash_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if stage in {"inspect", "prepare", "all"}:
        inventory = SleapGerbilsAdapter().inspect(raw)
        manifest["inventory"] = asdict(inventory)
        if stage in {"prepare", "all"}:
            observations = work / "prepared" / "sleap_gerbils" / "manifests" / "observations.jsonl"
            if not observations.is_file():
                bundle = SleapGerbilsAdapter().build_manifests(raw, work / "prepared" / "sleap_gerbils")
                manifest["prepared"] = {"observations": str(bundle.observations), "truth": str(bundle.private_eval_truth)}
            else:
                manifest["prepared"] = {"observations": str(observations), "status": "EXISTING_NOT_REBUILT"}
    if stage in {"runtime", "smoke", "predict", "eval", "clip", "all"}:
        env = _sleap_runtime_env(work)
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        backend = SleapNNBackend(executable=args.executable or str(work / "env" / "sleap_site" / "bin" / "sleap-nn"), device=args.device, env=env)
        if stage in {"runtime", "all"}:
            manifest["runtime"] = backend.verify_runtime()
            version = manifest["runtime"].get("version", {}).get("stdout_first_line")
            manifest["sleap_nn_version"] = version
            try:
                torch_probe = subprocess.run(
                    [_executable_python(backend.executable), "-c", "import torch; print(torch.__version__)"],
                    env=env, capture_output=True, text=True, timeout=30, check=False,
                )
                manifest["torch_version"] = torch_probe.stdout.strip() or None
            except Exception:
                manifest["torch_version"] = None
            try:
                sleap_io_probe = subprocess.run(
                    [_executable_python(backend.executable), "-c", "import sleap_io; print(getattr(sleap_io, '__version__', 'unknown'))"],
                    env=env, capture_output=True, text=True, timeout=30, check=False,
                )
                manifest["sleap_io_version"] = sleap_io_probe.stdout.strip() or None
            except Exception:
                manifest["sleap_io_version"] = None
        config_dir = work / "runs" / "sleap_gerbils_pose_config"
        config_path = config_dir / "training_config.yaml"
        checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
        if stage in {"smoke", "all"}:
            if checkpoint is None:
                candidates = sorted((work / "runs" / "sleap_gerbils_pose_smoke").rglob("best.ckpt"))
                checkpoint = candidates[0] if candidates else None
            if checkpoint is None:
                config_dir.mkdir(parents=True, exist_ok=True)
                generated = backend.generate_config(required["train"], config_dir)
                config_path = next((path for path in generated if path.name == "training_config.yaml"), generated[0])
                checkpoint = backend.train(config_path, required["train"], required["val"], work / "runs" / "sleap_gerbils_pose_smoke", args.max_epochs)
            manifest["pose_checkpoint"] = str(checkpoint)
        if checkpoint is None and stage != "runtime":
            manifest["blockers"].append("missing trained SLEAP checkpoint")
        else:
            manifest["pose_checkpoint"] = str(checkpoint)
            if checkpoint.is_file():
                manifest["pose_checkpoint_sha256"] = _sha256_file(checkpoint)
                manifest["pose_checkpoint_hash"] = manifest["pose_checkpoint_sha256"]
            if stage in {"predict", "eval", "all"}:
                prediction = run_dir / "test_predictions.slp"
                if not prediction.is_file():
                    backend.predict(required["test"], checkpoint, prediction, only_labeled_frames=True)
                manifest["test_prediction"] = str(prediction)
                if stage in {"eval", "all"}:
                    manifest["test_eval"] = backend.evaluate(required["test"], prediction, run_dir / "eval")
            if stage in {"clip", "all"}:
                clip_prediction = run_dir / "example_5min.predictions.slp"
                if not clip_prediction.is_file():
                    backend.predict(required["clip"], checkpoint, clip_prediction, tracking=True, frames=args.clip_frames)
                manifest["clip_prediction"] = str(clip_prediction)
    manifest.update(_training_stats(work))
    manifest.setdefault("identity_model_sha256", None)
    manifest["identity_model_hash"] = manifest.get("identity_model_sha256")
    manifest.setdefault("gallery_versions", [])
    # A successful upstream process is not the same as a measured baseline:
    # zero predicted instances and a frame-prefix clip are explicit partial
    # evaluation states, never silently promoted to metrics.
    if manifest.get("test_eval", {}).get("status") == "SUCCEEDED_NO_PREDICTIONS":
        manifest["blockers"].append("pose_evaluation_has_no_predicted_instances")
    if stage in {"clip", "all"} and args.clip_frames.strip() != "0-2559":
        manifest["blockers"].append("clip_tracking_is_prefix_smoke_only")
    manifest["blockers"] = sorted(set(manifest["blockers"]))
    manifest["status"] = "SUCCEEDED" if not manifest["blockers"] else "PARTIALLY_EVALUATED"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["end_time"] = manifest["finished_at"]
    _write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0 if manifest["status"] == "SUCCEEDED" else 2


def experiment_identity(args) -> int:
    """Start a B0/B1/B2/O1 run only with verified identity inputs."""
    work = _work_root(args.work_root)
    default_configs = {
        "b0": ROOT / "configs" / "experiments" / "gerbils" / "B0_global_static.yaml",
        "b1": ROOT / "configs" / "experiments" / "gerbils" / "B1_part_static.yaml",
        "b2": ROOT / "configs" / "experiments" / "gerbils" / "B2_learned_matcher.yaml",
        "o1": ROOT / "configs" / "experiments" / "gerbils" / "O1_safe_memory.yaml",
    }
    config = Path(args.config) if args.config else default_configs[args.experiment]
    run_dir = work / "runs" / f"{args.experiment}_gerbils"
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    checkpoint = Path(args.identity_checkpoint).expanduser().resolve() if args.identity_checkpoint else None
    result: dict[str, Any] = {"schema_version": "mat.identity_experiment.v1", "experiment": args.experiment, "config": str(config),
                              "command": "mat experiment " + args.experiment, "argv": [str(value) for value in sys.argv], "start_time": started_at,
                              "end_time": None, "hostname": platform.node(), "GPU": None,
                              "dataset_sha256": None, "split_sha256": None,
                              "sleap_io_version": None, "sleap_nn_version": None, "torch_version": None,
                              "identity_model_hash": None, "pose_checkpoint_hash": None,
                              "gallery_start_version": None, "gallery_end_version": None,
                              "status": "BLOCKED_MISSING_IDENTITY_ASSET", "optimizer_steps": None,
                              "checkpoint": str(checkpoint) if checkpoint else None, "metrics": None, "blockers": []}
    if not config.is_file():
        result["status"] = "BLOCKED_MISSING_CONFIG"; result["blockers"].append(str(config))
    elif checkpoint is None or not checkpoint.is_file():
        result["blockers"].append("verified MegaDescriptor-T-224 checkpoint/config not supplied")
    else:
        from mat.backends.wildlife import GlobalIdentityBackend
        model_config = checkpoint.with_name("config.json")
        try:
            GlobalIdentityBackend.from_local(checkpoint, config_path=model_config)
        except Exception as exc:
            result["status"] = "BLOCKED_IDENTITY_ASSET_VALIDATION"; result["blockers"].append(f"{type(exc).__name__}: {exc}")
        else:
            result["status"] = "NOT_RUN_MISSING_S0_INPUT"; result["blockers"].append("file-backed S0/query crop runner and verified identity mapping are not available")
    result["end_time"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "run_manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 2


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
    p.add_argument("--stage", choices=["inspect", "prepare", "runtime", "smoke", "predict", "eval", "clip", "all"], default="all")
    p.add_argument("--work-root")
    p.add_argument("--run-dir")
    p.add_argument("--executable")
    p.add_argument("--checkpoint")
    p.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    p.add_argument("--max-epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--clip-frames", default="0-15")
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
