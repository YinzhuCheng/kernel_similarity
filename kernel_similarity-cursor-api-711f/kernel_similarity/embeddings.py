import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    concurrency: int
    timeout: float
    cache_path: Optional[str] = None
    cache_only: bool = False


class OpenAIEmbeddingClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 8.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._connectivity_reported = False
        self._connectivity_lock = threading.Lock()

    def embed(self, inputs: List[str]) -> List[List[float]]:
        # OpenAI 兼容接口
        url = f"{self.api_base}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": inputs}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = [row["embedding"] for row in data["data"]]
                if len(embeddings) != len(inputs):
                    raise ValueError("Embedding count mismatch.")
                self._report_connectivity(True)
                return embeddings
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    self._report_connectivity(False, exc)
                    raise
                delay = min(
                    self.backoff_base * (2 ** (attempt - 1)), self.backoff_max
                )
                jitter = random.uniform(0, delay * 0.1)
                time.sleep(delay + jitter)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Embedding request failed with unknown error.")

    def _report_connectivity(self, success: bool, error: Optional[Exception] = None) -> None:
        with self._connectivity_lock:
            if self._connectivity_reported:
                return
            self._connectivity_reported = True
        if success:
            print(f"embedding api reachable: {self.api_base}")
        else:
            detail = f" ({error})" if error else ""
            print(f"embedding api unreachable: {self.api_base}{detail}")


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
        batches = list(chunked(missing, config.batch_size))
        concurrency = max(1, config.concurrency)

        def embed_batch(batch_indices: List[int]) -> Dict[int, List[float]]:
            batch_texts = [texts[i] for i in batch_indices]
            batch_embeddings = client.embed(batch_texts)
            return {
                row_idx: embedding
                for row_idx, embedding in zip(batch_indices, batch_embeddings)
            }

        first_batch = batches[:1]
        remaining_batches = batches[1:]
        if first_batch:
            new_embeddings.update(embed_batch(first_batch[0]))

        if remaining_batches:
            if concurrency == 1:
                for batch_indices in remaining_batches:
                    new_embeddings.update(embed_batch(batch_indices))
            else:
                worker_count = min(concurrency, len(remaining_batches))
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(embed_batch, batch_indices)
                        for batch_indices in remaining_batches
                    ]
                    for future in as_completed(futures):
                        new_embeddings.update(future.result())
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
