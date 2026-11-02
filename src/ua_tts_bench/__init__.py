"""Benchmark for Ukrainian text-to-speech models."""

from ua_tts_bench.evaluation import evaluate
from ua_tts_bench.generation import generate
from ua_tts_bench.model import TtsModel

__all__ = ["TtsModel", "evaluate", "generate"]
