"""Pure text helpers shared by embeddings, BM25, grading and the fake LLM."""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "this to was were what when where which who why will with does do did can".split()
)


def _stem(token: str) -> str:
    """Very light suffix stripping (lasts->last, robots->robot, charging->charg)."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, stopwords removed, light stemming."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def normalize_text(text: str) -> str:
    """Collapse whitespace noise while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Stable hash of normalised content (whitespace/case-insensitive)."""
    canon = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
