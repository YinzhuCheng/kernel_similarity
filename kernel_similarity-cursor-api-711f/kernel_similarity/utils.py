import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class RunConfig:
    seed: int
    device: str
    output_dir: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def flatten_pairs(pairs: Dict[str, List[Tuple[int, int]]]) -> List[Tuple[str, int, int]]:
    flat: List[Tuple[str, int, int]] = []
    for qid, rows in pairs.items():
        for doc_idx, label in rows:
            flat.append((qid, doc_idx, label))
    return flat
