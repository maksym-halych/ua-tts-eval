"""Conversion between float samples and WAV PCM16 mono."""

import io
import wave

import numpy as np
import soxr

PCM16_MAX = 32767


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Validate model output and mix it down to one channel."""
    audio = np.asarray(samples)
    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError(f"samples must be float, got {audio.dtype}")
    if audio.size == 0:
        raise ValueError("empty audio")
    if audio.ndim == 2:
        # Models return either (samples, channels) or (channels, samples). There are always fewer channels.
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if audio.ndim != 1:
        raise ValueError(f"samples must have 1 or 2 dimensions, got {audio.ndim}")
    if not np.isfinite(audio).all():
        raise ValueError("NaN or infinite samples")
    return audio


def count_clipped(audio: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(audio) > 1.0))


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float samples as WAV PCM16, clipping anything outside ±1."""
    pcm = np.round(np.clip(audio, -1.0, 1.0) * PCM16_MAX).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / PCM16_MAX, sample_rate


def resample(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    return soxr.resample(audio, sample_rate, target_rate)
