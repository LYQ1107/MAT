from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def configure_json_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mat")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def event(logger: logging.Logger, name: str, **payload: Any) -> None:
    logger.info(json.dumps({"event": name, **payload}, ensure_ascii=False, sort_keys=True, default=str))

