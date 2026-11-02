"""Text normalization and the error rates computed on normalized text."""

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

APOSTROPHES = str.maketrans({"ʼ": "'", "’": "'"})
# An apostrophe is part of a word only between two word characters. Elsewhere it is a quotation mark.
QUOTE_LIKE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")


def _clean_char(char: str) -> str:
    category = unicodedata.category(char)
    if category == "Pd":
        return " "
    if category.startswith("P") and char != "'":
        return ""
    return char


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().translate(APOSTROPHES)
    text = "".join(_clean_char(char) for char in text)
    text = QUOTE_LIKE_APOSTROPHE.sub("", text)
    return " ".join(text.split())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate of two normalized strings."""
    ref_words = reference.split()
    return Levenshtein.distance(ref_words, hypothesis.split()) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of two normalized strings, counting spaces."""
    return Levenshtein.distance(reference, hypothesis) / len(reference)
