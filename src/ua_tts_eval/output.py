"""Helpers for writing the output folder."""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

FORMAT_VERSION = 1
OUTPUT_ROOT = Path(".cache/ua-tts-eval")

_logger = logging.getLogger("ua_tts_eval")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data: dict) -> None:
    """Replace the file in one step, so an interrupted write can't leave a partial file."""
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def bench_log(folder: Path) -> Iterator[logging.Logger]:
    """Log to bench.log in the folder and to stderr, which shows progress during long runs."""
    formatter = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = time.gmtime
    handlers = [logging.FileHandler(folder / "bench.log", encoding="utf-8"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    try:
        yield _logger
    finally:
        for handler in handlers:
            _logger.removeHandler(handler)
            handler.close()
