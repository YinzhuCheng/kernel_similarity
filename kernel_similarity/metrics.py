from typing import Dict, Iterable, List


def recall_at_k(ranked: List[int], positives: Iterable[int], k: int) -> float:
    if k <= 0:
        return 0.0
    pos_set = set(positives)
    if not pos_set:
        return 0.0
    hit = any(doc_id in pos_set for doc_id in ranked[:k])
    return 1.0 if hit else 0.0


def mrr_at_k(ranked: List[int], positives: Iterable[int], k: int) -> float:
    pos_set = set(positives)
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in pos_set:
            return 1.0 / rank
    return 0.0


def aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    out: Dict[str, float] = {}
    for key in keys:
        out[key] = sum(item[key] for item in metrics) / len(metrics)
    return out
