import argparse
import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from .baselines import ContrastiveConfig, score_contrastive, train_contrastive
from .data import build_positive_map, load_corpus, load_qrels, load_queries, split_queries
from .embeddings import EmbeddingConfig, ensure_embeddings
from .gp import GPTrainingConfig, score_with_gp, train_multiquery_gp, train_single_query_gp, build_query_models
from .kernels import KernelConfig, build_weighted_kernel
from .metrics import aggregate_metrics, mrr_at_k, recall_at_k
from .retrieval import BM25Index, cosine_similarity_scores, top_n_cosine
from .utils import ensure_dir, save_json, set_seed, utc_timestamp


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _device_from_arg(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available, falling back to cpu")
        return "cpu"
    return device


def build_training_data(
    query_ids: List[str],
    positives: Dict[str, List[str]],
    doc_id_to_idx: Dict[str, int],
    doc_embeddings: np.ndarray,
    query_embeddings: Dict[str, np.ndarray],
    s_neg: int,
    s_pos: int,
    seed: int,
    device: str,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], List[Tuple[np.ndarray, np.ndarray, int]]]:
    rng = np.random.RandomState(seed)
    all_indices = np.arange(len(doc_embeddings))
    train_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    contrastive_pairs: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for qid in query_ids:
        pos_doc_ids = positives.get(qid, [])
        pos_indices = [doc_id_to_idx[doc_id] for doc_id in pos_doc_ids if doc_id in doc_id_to_idx]
        if not pos_indices:
            continue
        if s_pos > 0 and len(pos_indices) > s_pos:
            pos_indices = rng.choice(pos_indices, size=s_pos, replace=False).tolist()
        pos_set = set(pos_indices)
        neg_candidates = np.array([idx for idx in all_indices if idx not in pos_set])
        if len(neg_candidates) == 0:
            continue
        neg_count = min(s_neg, len(neg_candidates))
        neg_indices = rng.choice(neg_candidates, size=neg_count, replace=False).tolist()
        indices = pos_indices + neg_indices
        labels = [1] * len(pos_indices) + [0] * len(neg_indices)
        x = torch.tensor(doc_embeddings[indices], dtype=torch.float32, device=device)
        y = torch.tensor(labels, dtype=torch.float32, device=device)
        train_data[qid] = (x, y)
        if qid in query_embeddings:
            q_vec = query_embeddings[qid]
            for doc_idx, label in zip(indices, labels):
                contrastive_pairs.append((q_vec, doc_embeddings[doc_idx], label))
    return train_data, contrastive_pairs


def evaluate_rankings(
    rankings: Dict[str, List[int]],
    positives: Dict[str, List[int]],
    k_values: Tuple[int, int] = (10, 20),
) -> Dict[str, float]:
    metrics = []
    for qid, ranked in rankings.items():
        pos = positives.get(qid, [])
        metrics.append(
            {
                f"recall@{k_values[0]}": recall_at_k(ranked, pos, k_values[0]),
                f"recall@{k_values[1]}": recall_at_k(ranked, pos, k_values[1]),
                f"mrr@{k_values[0]}": mrr_at_k(ranked, pos, k_values[0]),
            }
        )
    return aggregate_metrics(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-kernel GP ranking")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--queries", required=True, help="Path to queries.jsonl")
    parser.add_argument("--qrels", required=True, help="Path to qrels.tsv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--s-neg", type=int, default=64)
    parser.add_argument("--s-pos", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs")

    parser.add_argument("--embedding-api-base", default=os.getenv("EMBEDDING_API_BASE", "https://api.openai.com"))
    parser.add_argument("--embedding-api-key", default=os.getenv("EMBEDDING_API_KEY", ""))
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--query-embedding-model", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-timeout", type=float, default=60.0)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--cache-only", action="store_true")

    parser.add_argument("--inducing-points", type=int, default=64)
    parser.add_argument("--inducing-init", choices=["random", "kmeans"], default="random")
    parser.add_argument("--batch-queries", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--test-epochs", type=int, default=5)
    parser.add_argument("--test-learning-rate", type=float, default=5e-3)

    parser.add_argument("--no-rbf", action="store_true")
    parser.add_argument("--no-matern", action="store_true")
    parser.add_argument("--no-poly", action="store_true")
    parser.add_argument("--matern-nu", type=float, default=2.5)
    parser.add_argument("--poly-degree", type=int, default=2)

    parser.add_argument("--candidate-method", choices=["bm25", "cosine"], default="bm25")
    parser.add_argument("--rerank-top-n", type=int, default=200)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)

    parser.add_argument("--contrastive-dim", type=int, default=256)
    parser.add_argument("--contrastive-temp", type=float, default=0.07)
    parser.add_argument("--contrastive-epochs", type=int, default=5)
    parser.add_argument("--contrastive-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-batch-size", type=int, default=128)

    args = parser.parse_args()

    device = _device_from_arg(args.device)
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    if not args.embedding_api_key and not args.cache_only:
        raise ValueError("Embedding API key is required unless --cache-only is set.")

    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    positives = build_positive_map(qrels)
    query_map = {q.query_id: q for q in queries}
    query_ids = [qid for qid in query_map.keys() if qid in positives]
    train_ids, test_ids = split_queries(query_ids, args.train_ratio, args.seed)

    print(f"docs={len(corpus)} queries={len(query_ids)} train={len(train_ids)} test={len(test_ids)}")

    doc_ids = [doc.doc_id for doc in corpus]
    doc_texts = [doc.full_text for doc in corpus]
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    cache_dir = args.cache_dir
    ensure_dir(cache_dir)
    doc_cache = os.path.join(cache_dir, f"doc_{_safe_name(args.embedding_model)}.npz")
    query_model = args.query_embedding_model or args.embedding_model
    query_cache = os.path.join(cache_dir, f"query_{_safe_name(query_model)}.npz")

    doc_config = EmbeddingConfig(
        api_base=args.embedding_api_base,
        api_key=args.embedding_api_key,
        model=args.embedding_model,
        batch_size=args.embedding_batch_size,
        timeout=args.embedding_timeout,
        cache_path=doc_cache,
        cache_only=args.cache_only,
    )
    query_config = EmbeddingConfig(
        api_base=args.embedding_api_base,
        api_key=args.embedding_api_key,
        model=query_model,
        batch_size=args.embedding_batch_size,
        timeout=args.embedding_timeout,
        cache_path=query_cache,
        cache_only=args.cache_only,
    )

    doc_embeddings = ensure_embeddings(doc_ids, doc_texts, doc_config, "document")
    query_ids_all = [q.query_id for q in queries]
    query_texts_all = [q.text for q in queries]
    query_embeddings_array = ensure_embeddings(
        query_ids_all, query_texts_all, query_config, "query"
    )
    query_embeddings = {
        qid: query_embeddings_array[idx] for idx, qid in enumerate(query_ids_all)
    }

    bm25_index = BM25Index.build(doc_texts, k1=args.bm25_k1, b=args.bm25_b)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1) + 1e-8

    train_data, contrastive_pairs = build_training_data(
        train_ids,
        positives,
        doc_id_to_idx,
        doc_embeddings,
        query_embeddings,
        args.s_neg,
        args.s_pos,
        args.seed,
        device,
    )
    if not train_data:
        raise RuntimeError("No training data constructed. Check qrels and sampling.")

    kernel_config = KernelConfig(
        use_rbf=not args.no_rbf,
        use_matern=not args.no_matern,
        use_poly=not args.no_poly,
        matern_nu=args.matern_nu,
        poly_degree=args.poly_degree,
    )
    kernel = build_weighted_kernel(kernel_config).to(device)
    gp_config = GPTrainingConfig(
        inducing_points=args.inducing_points,
        inducing_init=args.inducing_init,
        batch_queries=args.batch_queries,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    models = build_query_models(train_data, kernel, gp_config, args.seed)
    loss_log = train_multiquery_gp(train_data, models, kernel, gp_config)

    contrastive_config = ContrastiveConfig(
        projection_dim=args.contrastive_dim,
        temperature=args.contrastive_temp,
        epochs=args.contrastive_epochs,
        learning_rate=args.contrastive_lr,
        batch_size=args.contrastive_batch_size,
        device=device,
    )
    contrastive_model = train_contrastive(
        contrastive_pairs, doc_embeddings.shape[1], contrastive_config
    )

    test_pos_indices: Dict[str, List[int]] = {}
    for qid in test_ids:
        pos_doc_ids = positives.get(qid, [])
        pos_indices = [
            doc_id_to_idx[doc_id]
            for doc_id in pos_doc_ids
            if doc_id in doc_id_to_idx
        ]
        if pos_indices:
            test_pos_indices[qid] = pos_indices

    max_k = 20
    rankings_bm25: Dict[str, List[int]] = {}
    rankings_cosine: Dict[str, List[int]] = {}
    rankings_contrastive: Dict[str, List[int]] = {}
    rankings_gp: Dict[str, List[int]] = {}

    test_gp_config = GPTrainingConfig(
        inducing_points=args.inducing_points,
        inducing_init=args.inducing_init,
        batch_queries=1,
        epochs=args.test_epochs,
        learning_rate=args.test_learning_rate,
        device=device,
    )

    for qid in test_pos_indices:
        query_seed = args.seed + (abs(hash(qid)) % 10000)
        query_text = query_map[qid].text
        query_vec = query_embeddings[qid]

        bm25_top = bm25_index.top_n(query_text, max_k)
        rankings_bm25[qid] = bm25_top

        cosine_scores = cosine_similarity_scores(query_vec, doc_embeddings, doc_norms)
        cosine_top = cosine_scores.argsort()[::-1][:max_k].tolist()
        rankings_cosine[qid] = cosine_top

        contrast_scores = score_contrastive(
            contrastive_model, query_vec, doc_embeddings, device
        )
        contrast_top = contrast_scores.argsort()[::-1][:max_k].tolist()
        rankings_contrastive[qid] = contrast_top

        candidate_top_n = args.rerank_top_n
        if args.candidate_method == "bm25":
            candidate_indices = bm25_index.top_n(query_text, candidate_top_n)
        else:
            candidate_indices = top_n_cosine(
                query_vec, doc_embeddings, doc_norms, candidate_top_n
            )
        candidate_embeddings = torch.tensor(
            doc_embeddings[candidate_indices], dtype=torch.float32, device=device
        )
        test_kernel = copy.deepcopy(kernel).to(device)
        for param in test_kernel.parameters():
            param.requires_grad = False
        test_query_data, _ = build_training_data(
            [qid],
            positives,
            doc_id_to_idx,
            doc_embeddings,
            query_embeddings,
            args.s_neg,
            args.s_pos,
            query_seed,
            device,
        )
        if qid in test_query_data:
            x_test, y_test = test_query_data[qid]
            test_model = train_single_query_gp(
                x_test, y_test, test_kernel, test_gp_config, query_seed + 100
            )
            gp_scores = score_with_gp(test_model, candidate_embeddings).cpu().numpy()
            ranked_candidate = [
                candidate_indices[idx]
                for idx in gp_scores.argsort()[::-1].tolist()
            ]
        else:
            ranked_candidate = candidate_indices
        rankings_gp[qid] = ranked_candidate[:max_k]

    metrics_bm25 = evaluate_rankings(rankings_bm25, test_pos_indices)
    metrics_cosine = evaluate_rankings(rankings_cosine, test_pos_indices)
    metrics_contrastive = evaluate_rankings(rankings_contrastive, test_pos_indices)
    metrics_gp = evaluate_rankings(rankings_gp, test_pos_indices)

    run_id = utc_timestamp()
    summary = {
        "run_id": run_id,
        "train_queries": len(train_data),
        "test_queries": len(test_pos_indices),
        "kernel_components": kernel.kernel_summary(),
        "kernel_count": len(kernel.base_kernels),
        "metrics": {
            "bm25": metrics_bm25,
            "cosine": metrics_cosine,
            "contrastive": metrics_contrastive,
            "gp_multi_kernel": metrics_gp,
        },
        "rerank_top_n": args.rerank_top_n,
        "candidate_method": args.candidate_method,
        "train_loss": loss_log,
    }
    save_json(os.path.join(args.output_dir, f"summary_{run_id}.json"), summary)
    print("metrics:")
    print(summary["metrics"])


if __name__ == "__main__":
    main()
