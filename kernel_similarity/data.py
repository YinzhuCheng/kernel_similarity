import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .utils import ensure_dir, normalize_text, save_json


@dataclass
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        if self.title and self.text:
            return normalize_text(f"{self.title}. {self.text}")
        return normalize_text(self.title or self.text or "")


@dataclass
class Query:
    query_id: str
    text: str


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_corpus(path: str) -> List[Document]:
    docs: List[Document] = []
    for row in load_jsonl(path):
        docs.append(
            Document(
                doc_id=str(row["_id"]),
                title=str(row.get("title", "")),
                text=str(row.get("text", "")),
            )
        )
    return docs


def load_queries(path: str) -> List[Query]:
    queries: List[Query] = []
    for row in load_jsonl(path):
        queries.append(Query(query_id=str(row["_id"]), text=str(row.get("text", ""))))
    return queries


def load_qrels(path: str) -> Dict[str, List[Tuple[str, int]]]:
    qrels: Dict[str, List[Tuple[str, int]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(str(qid), []).append((str(doc_id), rel))
    return qrels


def build_positive_map(qrels: Dict[str, List[Tuple[str, int]]]) -> Dict[str, List[str]]:
    positives: Dict[str, List[str]] = {}
    for qid, entries in qrels.items():
        pos = [doc_id for doc_id, rel in entries if rel > 0]
        if pos:
            positives[qid] = pos
    return positives


def split_queries(
    query_ids: List[str], train_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    # 固定随机种子，生成稳定的训练/测试划分
    rng = random.Random(seed)
    shuffled = list(query_ids)
    rng.shuffle(shuffled)
    train_size = max(1, int(len(shuffled) * train_ratio))
    train_ids = shuffled[:train_size]
    test_ids = shuffled[train_size:]
    return train_ids, test_ids


def load_or_create_split(
    split_path: str, query_ids: List[str], train_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    # 如果已有划分文件就直接复用，保证后续运行一致
    if split_path and os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        train_ids = [qid for qid in data.get("train", []) if qid in query_ids]
        test_ids = [qid for qid in data.get("test", []) if qid in query_ids]
        if train_ids and test_ids:
            return train_ids, test_ids
    train_ids, test_ids = split_queries(query_ids, train_ratio, seed)
    if split_path:
        split_dir = os.path.dirname(split_path)
        if split_dir:
            ensure_dir(split_dir)
        save_json(
            split_path,
            {"train": train_ids, "test": test_ids, "seed": seed, "train_ratio": train_ratio},
        )
    return train_ids, test_ids
