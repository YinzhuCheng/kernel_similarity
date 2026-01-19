# kernel_similarity

Multi-kernel sparse GP ranking with per-query GPs and shared kernel learning.

## Features
- OpenAI-compatible `/v1/embeddings` client with configurable API base and key.
- Per-query sparse GP classification using document embeddings only.
- Shared multi-kernel parameters (RBF, Matern, Polynomial) learned on training queries.
- Baselines: BM25, cosine similarity, and a simple contrastive retriever.
- Metrics: `recall@10`, `recall@20`, and `mrr@10`.
- Rerank mode for candidate trimming (BM25 or cosine).

## Install
`pip install -r requirements.txt`

## Data format
- `corpus.jsonl`: `{_id, title, text}`
- `queries.jsonl`: `{_id, text}`
- `qrels.tsv`: `query_id<TAB>doc_id<TAB>relevance`

## Run (Windows-friendly)
Edit parameters in `kernel_similarity/config.py`, then start with:
- Open `run.py` and press Run in your IDE, or
- Double-click `run.bat` on Windows

Default training uses 20% of queries for kernel learning and 80% for evaluation. Each test query trains a new sparse GP with fixed kernel parameters and its own sampled positives/negatives, then reranks candidates by the GP posterior mean.

## Notes
- GP training defaults to CPU; set `TrainingSettings.device = "cuda"` to enable GPU when available.
- Use `cache_dir` and `cache_only` to reuse cached embeddings.
- Candidate reranking defaults to top 200 documents (`RetrievalSettings.rerank_top_n`).
- Use `ExperimentSettings.experiments` to run a single experiment (e.g., `["bm25"]`, `["kernel"]`, or `["ours"]`).
- Query splits are saved to `DataConfig.split_path` and reused on subsequent runs.
- Output summary is written to `runs/summary_<timestamp>.json` with kernel weights and metrics.
- Kernel components are configured by `KernelSettings` lists (`matern_nus`, `poly_degrees`, `poly_offsets`) and `rbf_kernels` count.
- Set `GPSettings.use_inducing_points = False` to disable inducing point approximation.