import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests

from .utils import chunked, ensure_dir


@dataclass
class EmbeddingConfig:
    api_base: str
    api_key: str
    model: str
    batch_size: int
    timeout: float
    cache_path: Optional[str] = None
    cache_only: bool = False


class OpenAIEmbeddingClient:
    def __init__(self, api_base: str, api_key: str, model: str, timeout: float = 60.0):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, inputs: List[str]) -> List[List[float]]:
        # OpenAI 兼容接口
        url = f"{self.api_base}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": inputs}
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        embeddings = [row["embedding"] for row in data["data"]]
        if len(embeddings) != len(inputs):
            raise ValueError("Embedding count mismatch.")
        return embeddings


def _load_cache(cache_path: str) -> Tuple[List[str], np.ndarray]:
    if not os.path.exists(cache_path):
        return [], np.empty((0, 0), dtype=np.float32)
    data = np.load(cache_path, allow_pickle=True)
    ids = data["ids"].tolist()
    embeddings = data["embeddings"]
    return ids, embeddings


def _save_cache(cache_path: str, ids: List[str], embeddings: np.ndarray) -> None:
    ensure_dir(os.path.dirname(cache_path))
    np.savez_compressed(cache_path, ids=np.array(ids, dtype=object), embeddings=embeddings)


def ensure_embeddings(
    ids: List[str],
    texts: List[str],
    config: EmbeddingConfig,
    label: str,
) -> np.ndarray:
    # 优先加载缓存，只补齐缺失的嵌入
    if len(ids) != len(texts):
        raise ValueError("IDs and texts size mismatch.")
    if not ids:
        return np.empty((0, 0), dtype=np.float32)
    cache_path = config.cache_path
    cached_ids: List[str] = []
    cached_embeddings = np.empty((0, 0), dtype=np.float32)
    if cache_path:
        cached_ids, cached_embeddings = _load_cache(cache_path)
    cached_map: Dict[str, int] = {doc_id: idx for idx, doc_id in enumerate(cached_ids)}
    missing = [idx for idx, doc_id in enumerate(ids) if doc_id not in cached_map]
    new_embeddings: Dict[int, List[float]] = {}
    if missing:
        if config.cache_only:
            raise RuntimeError(f"Missing {len(missing)} {label} embeddings in cache.")
        client = OpenAIEmbeddingClient(
            config.api_base, config.api_key, config.model, timeout=config.timeout
        )
        for batch_indices in chunked(missing, config.batch_size):
            batch_texts = [texts[i] for i in batch_indices]
            batch_embeddings = client.embed(batch_texts)
            for row_idx, embedding in zip(batch_indices, batch_embeddings):
                new_embeddings[row_idx] = embedding
    embedding_dim = (
        cached_embeddings.shape[1]
        if cached_embeddings.size
        else len(next(iter(new_embeddings.values())))
    )
    full_embeddings = np.zeros((len(ids), embedding_dim), dtype=np.float32)
    for i, doc_id in enumerate(ids):
        if doc_id in cached_map:
            full_embeddings[i] = cached_embeddings[cached_map[doc_id]]
        else:
            full_embeddings[i] = np.array(new_embeddings[i], dtype=np.float32)
    if cache_path and missing:
        merged_ids = list(cached_ids)
        merged_embeddings = (
            cached_embeddings.copy()
            if cached_embeddings.size
            else np.empty((0, embedding_dim), dtype=np.float32)
        )
        for idx in missing:
            merged_ids.append(ids[idx])
            merged_embeddings = np.vstack(
                [merged_embeddings, np.array(new_embeddings[idx], dtype=np.float32)]
            )
        _save_cache(cache_path, merged_ids, merged_embeddings)
    return full_embeddings
