# kernel_similarity

Multi-kernel sparse GP retrieval with per-query GPs and shared kernel learning.

## Entry point
`python run.py`

## Config
Edit parameters in `kernel_similarity-cursor-api-711f/kernel_similarity/config.py`.

## Install
`pip install -r requirements.txt`

## Data format
- `corpus.jsonl`: `{_id, title, text}`
- `queries.jsonl`: `{_id, text}`
- `qrels.tsv`: `query_id<TAB>doc_id<TAB>relevance`

## Notes
- GP training defaults to CPU; set `TrainingSettings.device = "cuda"` to enable GPU when available.
- Use `cache_dir` and `cache_only` to reuse cached embeddings.
- Query splits are saved to `DataConfig.split_path` and reused on subsequent runs.
- Output summary is written to `runs/summary_<timestamp>.json` with kernel weights and metrics.