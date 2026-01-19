import copy
import hashlib
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .baselines import ContrastiveConfig, score_contrastive, train_contrastive
from .config import RunConfig, get_default_config
from .data import (
    build_positive_map,
    load_corpus,
    load_or_create_split,
    load_qrels,
    load_queries,
)
from .embeddings import EmbeddingConfig, ensure_embeddings
from .gp import (
    GPTrainingConfig,
    build_query_models,
    score_with_gp,
    train_multiquery_gp,
    train_single_query_gp,
)
from .kernels import KernelConfig, build_weighted_kernel, kernel_similarity_scores
from .metrics import aggregate_metrics, mrr_at_k, recall_at_k
from .retrieval import BM25Index, cosine_similarity_scores, top_n_cosine
from .utils import ensure_dir, save_json, set_seed, utc_timestamp


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _device_from_config(device: str) -> str:
    # GPU 不可用时自动回退
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available, falling back to cpu")
        return "cpu"
    return device


def build_training_data(
    query_ids: List[str],
    positives: Dict[str, List[str]],
    doc_id_to_idx: Dict[str, int],
    doc_embeddings: np.ndarray,
    query_embeddings: Optional[Dict[str, np.ndarray]],
    s_neg: int,
    s_pos: int,
    seed: int,
    device: str,
    build_contrastive: bool,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], List[Tuple[np.ndarray, np.ndarray, int]]]:
    # 构造每个 query 的训练点（正负采样）
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
        if build_contrastive and query_embeddings and qid in query_embeddings:
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


EXPERIMENT_CHOICES: Set[str] = {"bm25", "cosine", "contrastive", "ours", "kernel"}


def _validate_experiments(experiments: List[str]) -> List[str]:
    # 仅允许指定实验，避免误跑
    if not experiments:
        raise ValueError("experiments 不能为空")
    for exp in experiments:
        if exp not in EXPERIMENT_CHOICES:
            raise ValueError(f"未知实验: {exp}")
    return experiments


def _needs_embeddings(experiments: Set[str], candidate_method: str) -> Tuple[bool, bool]:
    # 判断是否需要文档/查询嵌入
    needs_doc = bool(
        experiments.intersection({"cosine", "contrastive", "ours", "kernel"})
    )
    needs_query = bool(
        experiments.intersection({"cosine", "contrastive", "kernel"})
    )
    if "ours" in experiments and candidate_method == "cosine":
        needs_query = True
    return needs_doc, needs_query


def run(config: RunConfig) -> None:
    # 所有参数从 config 读取，不使用命令行
    device = _device_from_config(config.training.device)
    set_seed(config.training.seed)
    ensure_dir(config.data.output_dir)

    # 1) 加载数据并固定训练/测试划分
    corpus = load_corpus(config.data.corpus_path)
    queries = load_queries(config.data.queries_path)
    qrels = load_qrels(config.data.qrels_path)
    positives = build_positive_map(qrels)
    query_map = {q.query_id: q for q in queries}
    query_ids = [qid for qid in query_map.keys() if qid in positives]
    train_ids, test_ids = load_or_create_split(
        config.data.split_path, query_ids, config.training.train_ratio, config.training.seed
    )

    print(
        f"docs={len(corpus)} queries={len(query_ids)} train={len(train_ids)} test={len(test_ids)}"
    )

    experiments = set(_validate_experiments(config.experiments.experiments))
    needs_doc_embeddings, needs_query_embeddings = _needs_embeddings(
        experiments, config.retrieval.candidate_method
    )

    if (needs_doc_embeddings or needs_query_embeddings) and not config.embedding.api_key and not config.embedding.cache_only:
        raise ValueError("Embedding API key is required unless cache_only is set.")

    # 2) 文档与查询映射
    doc_ids = [doc.doc_id for doc in corpus]
    doc_texts = [doc.full_text for doc in corpus]
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    doc_embeddings = np.empty((0, 0), dtype=np.float32)
    query_embeddings: Dict[str, np.ndarray] = {}
    # 3) 嵌入（第一次计算后走缓存复用）
    if needs_doc_embeddings or needs_query_embeddings:
        ensure_dir(config.embedding.cache_dir)
        doc_cache = os.path.join(
            config.embedding.cache_dir, f"doc_{_safe_name(config.embedding.doc_model)}.npz"
        )
        query_model = config.embedding.query_model or config.embedding.doc_model
        query_cache = os.path.join(
            config.embedding.cache_dir, f"query_{_safe_name(query_model)}.npz"
        )
        doc_config = EmbeddingConfig(
            api_base=config.embedding.api_base,
            api_key=config.embedding.api_key,
            model=config.embedding.doc_model,
            batch_size=config.embedding.batch_size,
            concurrency=config.embedding.concurrency,
            timeout=config.embedding.timeout,
            cache_path=doc_cache,
            cache_only=config.embedding.cache_only,
        )
        query_config = EmbeddingConfig(
            api_base=config.embedding.api_base,
            api_key=config.embedding.api_key,
            model=query_model,
            batch_size=config.embedding.batch_size,
            concurrency=config.embedding.concurrency,
            timeout=config.embedding.timeout,
            cache_path=query_cache,
            cache_only=config.embedding.cache_only,
        )
        if needs_doc_embeddings:
            doc_embeddings = ensure_embeddings(
                doc_ids, doc_texts, doc_config, "document"
            )
        if needs_query_embeddings:
            query_ids_all = [q.query_id for q in queries]
            query_texts_all = [q.text for q in queries]
            query_embeddings_array = ensure_embeddings(
                query_ids_all, query_texts_all, query_config, "query"
            )
            query_embeddings = {
                qid: query_embeddings_array[idx]
                for idx, qid in enumerate(query_ids_all)
            }

    bm25_index = None
    if "bm25" in experiments or (
        "ours" in experiments and config.retrieval.candidate_method == "bm25"
    ):
        bm25_index = BM25Index.build(
            doc_texts, k1=config.retrieval.bm25_k1, b=config.retrieval.bm25_b
        )

    doc_norms = None
    if needs_doc_embeddings:
        doc_norms = np.linalg.norm(doc_embeddings, axis=1) + 1e-8

    train_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    contrastive_pairs: List[Tuple[np.ndarray, np.ndarray, int]] = []
    # 4) 训练数据构造（正负采样）
    if "ours" in experiments or "contrastive" in experiments or "kernel" in experiments:
        if doc_embeddings.size == 0:
            raise RuntimeError("Doc embeddings required for selected experiments.")
        train_data, contrastive_pairs = build_training_data(
            train_ids,
            positives,
            doc_id_to_idx,
            doc_embeddings,
            query_embeddings if needs_query_embeddings else None,
            config.training.s_neg,
            config.training.s_pos,
            config.training.seed,
            device,
            build_contrastive="contrastive" in experiments,
        )
        if not train_data:
            raise RuntimeError("No training data constructed. Check qrels and sampling.")

    kernel = None
    loss_log: Dict[str, float] = {}
    needs_kernel_training = "ours" in experiments or "kernel" in experiments
    # 5) 共享多核训练（ours/kernel）
    if needs_kernel_training:
        kernel_config = KernelConfig(
            rbf_kernels=config.kernel.rbf_kernels,
            matern_nus=config.kernel.matern_nus,
            poly_degrees=config.kernel.poly_degrees,
            poly_offsets=config.kernel.poly_offsets,
        )
        kernel = build_weighted_kernel(kernel_config).to(device)
        gp_config = GPTrainingConfig(
            inducing_points=config.gp.inducing_points,
            inducing_init=config.gp.inducing_init,
            batch_queries=config.gp.batch_queries,
            epochs=config.gp.epochs,
            learning_rate=config.gp.learning_rate,
            device=device,
            use_inducing_points=config.gp.use_inducing_points,
        )
        models = build_query_models(train_data, kernel, gp_config, config.training.seed)
        loss_log = train_multiquery_gp(train_data, models, kernel, gp_config)

    contrastive_model = None
    # 6) 对比学习 baseline
    if "contrastive" in experiments:
        contrastive_config = ContrastiveConfig(
            projection_dim=config.contrastive.projection_dim,
            temperature=config.contrastive.temperature,
            epochs=config.contrastive.epochs,
            learning_rate=config.contrastive.learning_rate,
            batch_size=config.contrastive.batch_size,
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

    rankings: Dict[str, Dict[str, List[int]]] = {}
    if "bm25" in experiments:
        rankings["bm25"] = {}
    if "cosine" in experiments:
        rankings["cosine"] = {}
    if "contrastive" in experiments:
        rankings["contrastive"] = {}
    if "ours" in experiments:
        rankings["ours"] = {}
    if "kernel" in experiments:
        rankings["kernel"] = {}

    test_gp_config = GPTrainingConfig(
        inducing_points=config.gp.inducing_points,
        inducing_init=config.gp.inducing_init,
        batch_queries=1,
        epochs=config.gp.test_epochs,
        learning_rate=config.gp.test_learning_rate,
        device=device,
        use_inducing_points=config.gp.use_inducing_points,
    )

    # 7) 逐 query 评测
    for qid in test_pos_indices:
        stable_hash = int(hashlib.md5(qid.encode("utf-8")).hexdigest()[:8], 16)
        query_seed = config.training.seed + (stable_hash % 10000)
        query_text = query_map[qid].text
        query_vec = query_embeddings.get(qid) if needs_query_embeddings else None

        if "bm25" in experiments and bm25_index:
            rankings["bm25"][qid] = bm25_index.top_n(query_text, config.retrieval.max_k)

        if "cosine" in experiments:
            if query_vec is None or doc_norms is None:
                raise RuntimeError("Cosine baseline requires query/doc embeddings.")
            cosine_scores = cosine_similarity_scores(
                query_vec, doc_embeddings, doc_norms
            )
            rankings["cosine"][qid] = (
                cosine_scores.argsort()[::-1][: config.retrieval.max_k].tolist()
            )

        if "kernel" in experiments:
            if kernel is None:
                raise RuntimeError("Kernel is required for kernel similarity.")
            if query_vec is None:
                raise RuntimeError("Kernel similarity requires query embeddings.")
            kernel_scores = kernel_similarity_scores(
                query_vec, doc_embeddings, kernel, device
            )
            rankings["kernel"][qid] = (
                kernel_scores.argsort()[::-1][: config.retrieval.max_k].tolist()
            )

        if "contrastive" in experiments:
            if contrastive_model is None or query_vec is None:
                raise RuntimeError("Contrastive baseline requires query embeddings.")
            contrast_scores = score_contrastive(
                contrastive_model, query_vec, doc_embeddings, device
            )
            rankings["contrastive"][qid] = (
                contrast_scores.argsort()[::-1][: config.retrieval.max_k].tolist()
            )

        if "ours" in experiments:
            if kernel is None:
                raise RuntimeError("Kernel is required for our method.")
            if config.retrieval.candidate_method == "bm25":
                if bm25_index is None:
                    raise RuntimeError("BM25 index required for candidate retrieval.")
                candidate_indices = bm25_index.top_n(
                    query_text, config.retrieval.rerank_top_n
                )
            else:
                if query_vec is None or doc_norms is None:
                    raise RuntimeError("Cosine candidates require query/doc embeddings.")
                candidate_indices = top_n_cosine(
                    query_vec,
                    doc_embeddings,
                    doc_norms,
                    config.retrieval.rerank_top_n,
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
                None,
                config.training.s_neg,
                config.training.s_pos,
                query_seed,
                device,
                build_contrastive=False,
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
            rankings["ours"][qid] = ranked_candidate[: config.retrieval.max_k]

    summary_metrics: Dict[str, Dict[str, float]] = {}
    for exp_name, ranked in rankings.items():
        summary_metrics[exp_name] = evaluate_rankings(
            ranked, test_pos_indices, config.retrieval.metrics_k
        )

    run_id = utc_timestamp()
    summary = {
        "run_id": run_id,
        "train_queries": len(train_data),
        "test_queries": len(test_pos_indices),
        "experiments": sorted(experiments),
        "candidate_method": config.retrieval.candidate_method,
        "rerank_top_n": config.retrieval.rerank_top_n,
        "kernel_components": kernel.kernel_summary() if kernel else [],
        "kernel_count": len(kernel.base_kernels) if kernel else 0,
        "metrics": summary_metrics,
        "train_loss": loss_log,
        "split_path": config.data.split_path,
    }
    save_json(os.path.join(config.data.output_dir, f"summary_{run_id}.json"), summary)
    print("metrics:")
    print(summary_metrics)


def main() -> None:
    # 入口函数：修改 config 即可调整全部参数
    config = get_default_config()
    run(config)


if __name__ == "__main__":
    main()
