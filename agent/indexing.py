"""Local vector index: TF-IDF embeddings with cosine similarity.

Design choice (documented in the README): the corpus is 14 small documents, so
a deterministic in-process sparse-vector embedding gives reliable retrieval
with zero infrastructure and fully reproducible evaluation. The ``VectorIndex``
interface is deliberately simple so a hosted embedding model can be swapped in
without touching the rest of the pipeline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .contracts import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "its",
    "may", "me", "my", "of", "on", "or", "our", "should", "so", "the", "their",
    "them", "there", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "would", "you", "your",
}


def _stem(token: str) -> str:
    """Minimal deterministic suffix normalization.

    Keeps plural/verb-form variants comparable (ships->ship,
    internationally->international, estimates->estimate) without a real
    stemming library.
    """
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if token.endswith("ly") and len(token) > 6:
        return token[:-2]
    if token.endswith("ing") and len(token) > 6:
        base = token[:-3]
        if len(base) >= 3 and base[-1] == base[-2]:  # shipping -> shipp -> ship
            base = base[:-1]
        return base
    return token


# Small domain synonym table: canonicalizes word forms that suffix stripping
# cannot unify (bug diary: 'canadian' never matched 'canada', hiding the
# delivery-estimate section for Canada questions).
_ALIASES = {
    "canadian": "canada",
    "dishwashers": "dishwasher",
    "tumblers": "tumbler",
}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    kept = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    stemmed = [_stem(t) for t in kept]
    return [_ALIASES.get(t, t) for t in stemmed]


@dataclass
class _Entry:
    chunk: Chunk
    tf: Counter
    norm: float


class VectorIndex:
    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._df: Counter = Counter()
        self._built = False

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tf = Counter(tokenize(chunk.text))
            # Heading tokens are strong intent signals; count them twice.
            tf.update(tokenize(" ".join(chunk.heading_path)))
            self._entries.append(_Entry(chunk=chunk, tf=tf, norm=0.0))
        self._built = False

    def build(self) -> None:
        self._df = Counter()
        for e in self._entries:
            for term in e.tf:
                self._df[term] += 1
        n = max(len(self._entries), 1)
        for e in self._entries:
            norm_sq = 0.0
            for term, freq in e.tf.items():
                w = self._weight(term, freq, n)
                norm_sq += w * w
            e.norm = math.sqrt(norm_sq) or 1.0
        self._built = True

    def _idf(self, term: str) -> float:
        n = max(len(self._entries), 1)
        return math.log((n + 1) / (self._df.get(term, 0) + 0.5))

    def _weight(self, term: str, freq: int, n: int) -> float:
        # Sub-linear TF dampening plus smooth IDF.
        return (1.0 + math.log(freq)) * self._idf(term)

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks by cosine similarity to the query."""
        if not self._built:
            self.build()
        q_tf = Counter(tokenize(query))
        # Order IDs like ORD-1007 are high-signal exact tokens.
        q_norm_sq = 0.0
        weights: dict[str, float] = {}
        for term, freq in q_tf.items():
            w = self._weight(term, freq, len(self._entries)) * (
                2.5 if re.fullmatch(r"ord\d+", term) else 1.0
            )
            weights[term] = w
            q_norm_sq += w * w
        q_norm = math.sqrt(q_norm_sq) or 1.0

        scored: list[tuple[Chunk, float]] = []
        for e in self._entries:
            dot = 0.0
            for term, qw in weights.items():
                freq = e.tf.get(term)
                if not freq:
                    continue
                dot += qw * self._weight(term, freq, len(self._entries))
            denom = q_norm * e.norm
            scored.append((e.chunk, dot / denom))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
