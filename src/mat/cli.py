from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from typing import Any

from mat import __version__
from mat.assets import AssetCatalog, AssetDownloader, DirectOnlyPolicy
from mat.assets.catalog import AssetSpec
from mat.data.adapters import RatIDAdapter, PigReIDAdapter, PigTrackingAdapter, MultiCamCowsAdapter
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


def assets_plan(args) -> int:
    catalog = load_catalog(args.catalog)
    specs = [asdict(a) for a in catalog.for_phase(args.phase)]
    work = _work_root(getattr(args, "work_root", None)); out = Path(args.output) if args.output else work / "assets" / f"asset_plan_{args.phase}.json"
    plan = {"schema_version": "mat.asset_plan.v1", "phase": args.phase, "route_status": "DIRECT_ROUTE_UNVERIFIED", "bulk_status": "BLOCKED_DIRECT_ROUTE", "assets": specs}
    _write_json(out, plan); print(out); return 0


def assets_fetch(args) -> int:
    raw = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    work = _work_root(args.work_root)
    downloader = AssetDownloader(work / "assets")
    policy = DirectOnlyPolicy(route_status="DIRECT_ROUTE_UNVERIFIED")
    receipts = []
    for item in raw.get("assets", []):
        asset = AssetSpec(**item)
        receipt = downloader.fetch(asset, policy)
        receipts.append(asdict(receipt))
    out = work / "assets" / "fetch_receipts.json"
    _write_json(out, {"schema_version": "mat.receipts.v1", "receipts": receipts}); print(out)
    return 0 if all(r.get("status") == "VERIFIED" for r in receipts) else 2


def assets_import(args) -> int:
    catalog = load_catalog(args.catalog); asset = catalog.get(args.asset)
    receipt = AssetDownloader(_work_root(args.work_root) / "assets").import_local(asset, Path(args.source))
    out = _work_root(args.work_root) / "assets" / "receipts" / f"{args.asset}.json"; _write_json(out, asdict(receipt)); print(json.dumps(asdict(receipt), ensure_ascii=False, indent=2)); return 0 if receipt.status == "VERIFIED" else 2


ADAPTERS = {"rat_id": RatIDAdapter, "pig_reid": PigReIDAdapter, "pig_tracking": PigTrackingAdapter, "multicamcows": MultiCamCowsAdapter}


def _adapter(name: str):
    if name not in ADAPTERS: raise ValueError(f"unknown dataset {name}")
    return ADAPTERS[name]()


def data_inspect(args) -> int:
    work = _work_root(args.work_root); raw = Path(args.raw_root) if args.raw_root else work / "assets" / "raw" / args.dataset
    adapter = _adapter(args.dataset)
    if not raw.is_dir():
        result = {"dataset": args.dataset, "status": "BLOCKED_MISSING_ASSET", "raw_root": str(raw), "reason": "authorized raw directory is absent"}
    else:
        result = asdict(adapter.inspect(raw)); result["status"] = "SMOKE_PASSED"
    out = work / "assets" / "manifests" / args.dataset / "inventory.json"; _write_json(out, result); print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


def data_prepare(args) -> int:
    work = _work_root(args.work_root); raw = Path(args.raw_root) if args.raw_root else work / "assets" / "raw" / args.dataset
    if not raw.is_dir():
        print(json.dumps({"status": "BLOCKED_MISSING_ASSET", "raw_root": str(raw)}, ensure_ascii=False)); return 2
    bundle = _adapter(args.dataset).build_manifests(raw, work / "assets" / "manifests" / args.dataset)
    print(json.dumps({"status": "SMOKE_PASSED", "observations": str(bundle.observations), "truth": str(bundle.private_eval_truth)}, ensure_ascii=False, indent=2)); return 0


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
    import yaml
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    validate_protocol(raw.get("protocol", {})); print("SMOKE_PASSED"); return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mat", description="MAT auditable longitudinal identity pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--work-root"); p.set_defaults(func=doctor)
    p = sub.add_parser("config-validate"); p.add_argument("--config", required=True); p.set_defaults(func=config_validate)
    sources = sub.add_parser("sources").add_subparsers(dest="sources_cmd", required=True)
    p = sources.add_parser("audit"); p.add_argument("--config", required=True); p.set_defaults(func=lambda a: (print(Path(a.config).read_text(encoding="utf-8")), 0)[1])
    assets = sub.add_parser("assets").add_subparsers(dest="assets_cmd", required=True)
    p = assets.add_parser("plan"); p.add_argument("--catalog", required=True); p.add_argument("--phase", choices=["P0", "P1", "P2", "P3"], required=True); p.add_argument("--work-root"); p.add_argument("--output"); p.set_defaults(func=assets_plan)
    p = assets.add_parser("fetch"); p.add_argument("--plan", required=True); p.add_argument("--work-root"); p.add_argument("--direct-only", action="store_true", required=True); p.set_defaults(func=assets_fetch)
    p = assets.add_parser("import"); p.add_argument("--asset", required=True); p.add_argument("--source", required=True); p.add_argument("--catalog"); p.add_argument("--work-root"); p.set_defaults(func=assets_import)
    data = sub.add_parser("data").add_subparsers(dest="data_cmd", required=True)
    p = data.add_parser("inspect"); p.add_argument("--dataset", choices=sorted(ADAPTERS), required=True); p.add_argument("--work-root"); p.add_argument("--raw-root"); p.set_defaults(func=data_inspect)
    p = data.add_parser("prepare"); p.add_argument("--dataset", choices=sorted(ADAPTERS), required=True); p.add_argument("--work-root"); p.add_argument("--raw-root"); p.set_defaults(func=data_prepare)
    split = sub.add_parser("split").add_subparsers(dest="split_cmd", required=True)
    p = split.add_parser("freeze"); p.add_argument("--manifest", required=True); p.add_argument("--field", default="cohort_uid"); p.add_argument("--seed", type=int, default=17); p.add_argument("--output"); p.add_argument("--work-root"); p.set_defaults(func=split_freeze)
    p = sub.add_parser("features").add_subparsers(dest="features_cmd", required=True).add_parser("extract"); p.add_argument("--config", required=True); p.add_argument("--scope", choices=["pilot", "full"], default="pilot"); p.add_argument("--work-root"); p.set_defaults(func=lambda a: blocked_stage(type("Args", (), {"stage":"features", "work_root":a.work_root})()))
    p = sub.add_parser("train").add_subparsers(dest="train_cmd", required=True).add_parser("source"); p.add_argument("--config", required=True); p.add_argument("--scope", choices=["pilot", "full"], default="pilot"); p.add_argument("--seed", type=int, default=17); p.add_argument("--work-root"); p.set_defaults(func=train_source)
    for name, command in [("reference", "export"), ("enroll", None), ("run", "cohort"), ("evaluate", None)]:
        parent = sub.add_parser(name)
        if command: parent = parent.add_subparsers(dest=f"{name}_cmd", required=True).add_parser(command)
        parent.add_argument("--config", required=False); parent.add_argument("--work-root"); parent.set_defaults(func=lambda a, s=name: blocked_stage(type("Args", (), {"stage":s, "work_root":a.work_root})()))
    return parser


def main(argv=None) -> int:
    parser = build_parser(); args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MATError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
