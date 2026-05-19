from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Set

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARABIC_VARIANTS = {
    "ي": "ی",
    "ك": "ک",
    "ۀ": "ه",
    "ة": "ه",
}


def load_stopwords(path: str | Path) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        words = {line.strip() for line in f if line.strip()}
    custom = {
        "این", "آن", "برای", "اما", "اگر", "است", "بود", "شد", "های", "یک",
        "هم", "را", "با", "که", "در", "از", "به", "و", "یا", "تا"
    }
    return words | custom


def normalize_persian_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text)
    for src, dst in ARABIC_VARIANTS.items():
        text = text.replace(src, dst)
    text = text.translate(PERSIAN_DIGITS)
    text = re.sub(r"[\[\]\(\){}\"'`,؛:!?؟|]+", " ", text)
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> list[str]:
    text = normalize_persian_text(text)
    return re.findall(r"[\w\u0600-\u06FF]+", text)


def remove_stopwords(tokens: Iterable[str], stopwords: Set[str]) -> list[str]:
    return [token for token in tokens if token not in stopwords and len(token) > 1]


def preprocess_text(text: object, stopwords: Set[str] | None = None) -> str:
    tokens = simple_tokenize(text)
    if stopwords is not None:
        tokens = remove_stopwords(tokens, stopwords)
    return " ".join(tokens)
