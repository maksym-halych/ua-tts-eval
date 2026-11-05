"""Access to the dataset shipped inside the package."""

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class Dataset:
    md5: str
    items: dict[str, dict]


def load_dataset() -> Dataset:
    raw = files("ua_tts_eval").joinpath("data/dataset.jsonl").read_bytes()
    items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return Dataset(hashlib.md5(raw).hexdigest(), {item["id"]: item for item in items})
