import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _tokenize(text: str) -> List[str]:
    return [tok for tok in text.lower().split() if tok]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    vocab_df: Dict[str, int] = None
    doc_freqs: List[Dict[str, int]] = None
    doc_lengths: List[int] = None
    avg_dl: float = 0.0

    @classmethod
    def build(cls, docs: Iterable[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        vocab_df: Dict[str, int] = {}
        doc_freqs: List[Dict[str, int]] = []
        doc_lengths: List[int] = []
        for text in docs:
            tokens = _tokenize(text)
            freq: Dict[str, int] = {}
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
            doc_freqs.append(freq)
            doc_lengths.append(len(tokens))
            for tok in freq:
                vocab_df[tok] = vocab_df.get(tok, 0) + 1
        avg_dl = float(np.mean(doc_lengths)) if doc_lengths else 0.0
        return cls(k1=k1, b=b, vocab_df=vocab_df, doc_freqs=doc_freqs, doc_lengths=doc_lengths, avg_dl=avg_dl)

    def score(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        scores = np.zeros(len(self.doc_freqs), dtype=np.float32)
        n_docs = len(self.doc_freqs)
        for tok in tokens:
            df = self.vocab_df.get(tok, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for i, freq in enumerate(self.doc_freqs):
                tf = freq.get(tok, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_dl + 1e-8))
                scores[i] += idf * (tf * (self.k1 + 1) / (denom + 1e-8))
        return scores

    def top_n(self, query: str, n: int) -> List[int]:
        scores = self.score(query)
        if n >= len(scores):
            return scores.argsort()[::-1].tolist()
        idx = np.argpartition(scores, -n)[-n:]
        ranked = idx[np.argsort(scores[idx])[::-1]]
        return ranked.tolist()


def cosine_similarity_scores(
    query_vec: np.ndarray, doc_matrix: np.ndarray, doc_norms: np.ndarray
) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec) + 1e-8
    scores = doc_matrix @ query_vec
    scores = scores / (doc_norms * query_norm + 1e-8)
    return scores


