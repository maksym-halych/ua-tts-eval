"""Make the MP3 copies that the arena plays."""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from ua_tts_bench.output import bench_log

TARGET_LUFS = -23.0
MAX_TRUE_PEAK_DBTP = -1.0
SAMPLE_RATE = 44100
BITRATE = "192k"
# ebur128 reports this level for audio it can't measure.
SILENCE_LUFS = -70.0
LOOP_SECONDS = "3"

_LOUDNESS = re.compile(r"Integrated loudness:\s*I:\s*(-?[\d.]+|-?inf|nan) LUFS", re.S)
_TRUE_PEAK = re.compile(r"True peak:\s*Peak:\s*(-?[\d.]+|-?inf|nan) dBFS", re.S)


def _ffmpeg(*args: str) -> str:
    """Run ffmpeg and return its log, which is where the measurements are."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffmpeg failed")
    return result.stderr


def _ebur128(*source: str) -> tuple[float, float] | None:
    log = _ffmpeg(*source, "-af", "ebur128=peak=true", "-f", "null", "-")
    summary = log[log.rindex("Summary:") :]
    loudness = float(_LOUDNESS.search(summary).group(1))
    true_peak = float(_TRUE_PEAK.search(summary).group(1))
    if not math.isfinite(loudness) or not math.isfinite(true_peak) or loudness <= SILENCE_LUFS:
        return None
    return loudness, true_peak


def _measure(wav: Path) -> tuple[float, float] | None:
    """Return (loudness in LUFS, true peak in dBTP), or None if the clip is silent."""
    # ebur128 needs 400 ms of audio. Repeating a shorter clip gives the level of its content.
    return _ebur128("-i", str(wav)) or _ebur128("-stream_loop", "-1", "-t", LOOP_SECONDS, "-i", str(wav))


def _encode(wav: Path, mp3: Path, gain_db: float) -> None:
    _ffmpeg(
        "-y",
        "-i", str(wav),
        "-af", f"volume={gain_db:.2f}dB",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "libmp3lame",
        "-b:a", BITRATE,
        "-map_metadata", "-1",
        str(mp3),
    )  # fmt: skip


def export_mp3(folder: Path) -> None:
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found, generation did not finish")
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    target = folder / "mp3"
    temporary = folder / "mp3.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()

    ok = skipped = failed = 0
    with bench_log(folder) as log:
        for item_id in manifest["items"]:
            try:
                wav = folder / "audio" / f"{item_id}.wav"
                measured = _measure(wav)
                if measured is None:
                    log.warning("export_mp3 %s silent, skipped", item_id)
                    skipped += 1
                    continue
                loudness, true_peak = measured
                # One constant gain, capped so that a clip ends up quieter than the target instead of distorted.
                gain_db = min(TARGET_LUFS - loudness, MAX_TRUE_PEAK_DBTP - true_peak)
                _encode(wav, temporary / f"{item_id}.mp3", gain_db)
            except Exception as error:
                log.error("export_mp3 %s %s: %s", item_id, type(error).__name__, error)
                failed += 1
                continue
            ok += 1

        log.info("export_mp3 done: %d ok, %d skipped, %d failed", ok, skipped, failed)

    # Nothing succeeded. The earlier copies are still the better record.
    if ok == 0:
        shutil.rmtree(temporary)
        raise RuntimeError(f"no items converted in {folder}, see bench.log. mp3/ was not changed")

    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)
