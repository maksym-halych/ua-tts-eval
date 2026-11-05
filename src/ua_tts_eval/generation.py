"""Generate audio for every dataset item with a TTS model."""

import json
import re
import shutil
import sys
import time
from pathlib import Path

from ua_tts_eval.audio import count_clipped, to_mono, to_wav_bytes
from ua_tts_eval.dataset import load_dataset
from ua_tts_eval.model import TtsModel
from ua_tts_eval.output import (
    FORMAT_VERSION,
    OUTPUT_ROOT,
    bench_log,
    utc_now,
    write_json,
)
from ua_tts_eval.provenance import hardware, model_source, package_commit

# The first character excludes "." and "..", which would make generate delete a parent folder.
VALID_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def generate(model: TtsModel) -> Path:
    if not VALID_NAME.fullmatch(model.name):
        raise ValueError(
            f"invalid model name {model.name!r}: use letters, digits, '.', '_', '-'"
        )
    # A problem here would otherwise only surface after the whole dataset is synthesized.
    json.dumps(model.params)

    # Read before the folder is deleted: the script may be a saved adapter.py running from that folder.
    script = Path(sys._getframe(1).f_code.co_filename).resolve()
    adapter = script.read_bytes() if script.is_file() else None
    source = model_source(script.parent if adapter is not None else Path.cwd())

    dataset = load_dataset()
    manifest = {
        "format_version": FORMAT_VERSION,
        "generated_at": utc_now(),
        "dataset_md5": dataset.md5,
        "ua-tts-eval": {"commit": package_commit()},
        "model": {**source, "params": model.params},
        "device": model.device,
        "hardware": hardware(),
        "items": {},
    }

    folder = Path.cwd() / OUTPUT_ROOT / model.name
    if folder.exists():
        shutil.rmtree(folder)
    audio_folder = folder / "audio"
    audio_folder.mkdir(parents=True)

    with bench_log(folder) as log:
        if adapter is not None:
            (folder / "adapter.py").write_bytes(adapter)
        else:
            log.warning("generate not called from a script file, adapter.py not saved")

        for item_id, item in dataset.items.items():
            path = audio_folder / f"{item_id}.wav"
            try:
                start = time.perf_counter()
                samples, sample_rate = model.synthesize(item["text"])
                elapsed = time.perf_counter() - start
                audio = to_mono(samples)
                path.write_bytes(to_wav_bytes(audio, sample_rate))
            except Exception as error:
                path.unlink(missing_ok=True)
                log.error("generate %s %s: %s", item_id, type(error).__name__, error)
                continue
            clipped = count_clipped(audio)
            if clipped:
                log.warning("generate %s clipped %d samples", item_id, clipped)
            log.info("generate %s ok %.2fs", item_id, elapsed)
            manifest["items"][item_id] = {"synthesis_time_s": round(elapsed, 3)}

        ok = len(manifest["items"])
        log.info("generate done: %d ok, %d failed", ok, len(dataset.items) - ok)

    write_json(folder / "manifest.json", manifest)
    return folder
