"""Compute metrics for a folder of generated audio."""

import json
import os
from pathlib import Path

from openai import OpenAI

from ua_tts_eval.audio import read_wav, resample, to_wav_bytes
from ua_tts_eval.dataset import load_dataset
from ua_tts_eval.output import FORMAT_VERSION, bench_log, utc_now, write_json
from ua_tts_eval.provenance import package_commit
from ua_tts_eval.text import cer, normalize, wer

ASR_SAMPLE_RATE = 16000


def _evaluate_item(client: OpenAI, asr_model: str, path: Path, reference: str, synthesis_time_s: float) -> dict:
    audio, sample_rate = read_wav(path)
    duration_s = len(audio) / sample_rate
    wav = to_wav_bytes(resample(audio, sample_rate, ASR_SAMPLE_RATE), ASR_SAMPLE_RATE)
    transcript = client.audio.transcriptions.create(
        model=asr_model,
        file=(path.name, wav, "audio/wav"),
        language="uk",
    ).text
    reference, hypothesis = normalize(reference), normalize(transcript)
    return {
        "transcript": transcript,
        "wer": round(wer(reference, hypothesis), 4),
        "cer": round(cer(reference, hypothesis), 4),
        "duration_s": round(duration_s, 3),
        "rtf": round(synthesis_time_s / duration_s, 3),
    }


def evaluate(folder: Path, asr_base_url: str, asr_model: str) -> None:
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found, generation did not finish")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dataset = load_dataset()
    client = OpenAI(base_url=asr_base_url, api_key=os.environ.get("OPENAI_API_KEY") or "unused")
    evaluated_at = utc_now()
    items = {}

    with bench_log(folder) as log:
        for item_id, generated in manifest["items"].items():
            try:
                items[item_id] = _evaluate_item(
                    client,
                    asr_model,
                    folder / "audio" / f"{item_id}.wav",
                    dataset.items[item_id]["text_normalized"],
                    generated["synthesis_time_s"],
                )
            except Exception as error:
                log.error("evaluate %s %s: %s", item_id, type(error).__name__, error)
                continue
            log.info("evaluate %s wer=%.3f cer=%.3f", item_id, items[item_id]["wer"], items[item_id]["cer"])

        log.info("evaluate done: %d ok, %d failed", len(items), len(manifest["items"]) - len(items))

    # Nothing succeeded, for example the ASR is down. Earlier metrics are still the better record.
    if not items:
        raise RuntimeError(f"no items evaluated in {folder}, see bench.log. metrics.json was not changed")

    write_json(
        folder / "metrics.json",
        {
            "format_version": FORMAT_VERSION,
            "evaluated_at": evaluated_at,
            "dataset_md5": dataset.md5,
            "ua-tts-eval": {"commit": package_commit()},
            "asr": asr_model,
            "items": items,
        },
    )
