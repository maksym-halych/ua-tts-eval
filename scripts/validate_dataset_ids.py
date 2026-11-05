"""Check that every item id in the dataset equals the md5 hash of its text."""

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ID_LENGTH = 12


def text_id(text: str) -> str:
    canonical = unicodedata.normalize("NFC", text.strip())
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()[:ID_LENGTH]


def validate(path: Path) -> list[str]:
    errors = []
    seen = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id, text = item.get("id"), item.get("text")
            if not text:
                errors.append(f"line {line_no}: missing text")
                continue
            expected = text_id(text)
            if item_id != expected:
                errors.append(f"line {line_no}: id {item_id!r} does not match text hash {expected!r} ({text[:50]!r})")
            if expected in seen:
                errors.append(f"line {line_no}: duplicate text, same as line {seen[expected]}")
            seen.setdefault(expected, line_no)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="src/ua_tts_eval/data/dataset.jsonl", type=Path)
    args = parser.parse_args()

    errors = validate(args.path)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(f"{args.path}: {len(errors)} id error(s)")
    print(f"{args.path}: all ids are valid")


if __name__ == "__main__":
    main()
