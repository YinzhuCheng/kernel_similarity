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

## Run
Example (SciFact layout):
`python -m kernel_similarity --corpus data/corpus.jsonl --queries data/queries.jsonl --qrels data/qrels.tsv --embedding-api-base https://api.openai.com --embedding-api-key $EMBEDDING_API_KEY`

Default training uses 20% of queries for kernel learning and 80% for evaluation. Each test query trains a new sparse GP with fixed kernel parameters and its own sampled positives/negatives, then reranks candidates by the GP posterior mean.

## Notes
- GP training defaults to CPU; use `--device cuda` to enable GPU when available.
- Use `--cache-dir` and `--cache-only` to reuse cached embeddings.
- Candidate reranking defaults to top 200 documents (`--rerank-top-n`).
- Output summary is written to `runs/summary_<timestamp>.json` with kernel weights and metrics.