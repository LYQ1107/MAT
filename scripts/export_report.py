#!/usr/bin/env python3
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Export only existing MAT run manifests")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.runs.rglob("*.json")):
        try: rows.append({"path": str(path), "manifest": json.loads(path.read_text(encoding="utf-8"))})
        except Exception: continue
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "mat.report.v1", "runs": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__": main()

