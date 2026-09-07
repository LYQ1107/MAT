#!/usr/bin/env python3
"""Audit-only source bootstrap; never clones with LFS or downloads model/data assets."""
from pathlib import Path
import json
import subprocess


def main():
    lock = Path(__file__).resolve().parents[1] / "locks/upstream.lock.yaml"
    print(json.dumps({"status": "AUDIT_ONLY", "lock": str(lock), "lfs_smudge": "disabled", "bulk_download": "blocked_until_route_approved"}))


if __name__ == "__main__":
    main()

