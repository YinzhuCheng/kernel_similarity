import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    # 数据路径与输出目录，统一在这里修改
    corpus_path: str = "data/corpus.jsonl"
    queries_path: str = "data/queries.jsonl"
    qrels_path: str = "data/qrels.tsv"
    output_dir: str = "runs"
    split_path: str = "runs/query_split.json"


@dataclass
class EmbeddingSettings:
    # 嵌入接口配置（OpenAI 协议兼容）
    api_base: str = "https://api.openai.com"
    api_key: str = ""
    doc_model: str = "text-embedding-3-small"
    query_model: str = ""
    batch_size: int = 32
    timeout: float = 60.0
    cache_dir: str = "cache"
    cache_only: bool = False


@dataclass
class TrainingSettings:
    # 训练/采样控制参数
    seed: int = 42
    train_ratio: float = 0.2
    s_neg: int = 64
    s_pos: int = 0
    device: str = "cpu"


@dataclass
class GPSettings:
    # 稀疏 GP 训练参数（共享多核）
    inducing_points: int = 64
    inducing_init: str = "random"  # random / kmeans
    batch_queries: int = 4
    epochs: int = 10
    learning_rate: float = 1e-2
    test_epochs: int = 5
    test_learning_rate: float = 5e-3


@dataclass
class KernelSettings:
    # 多核配置
    use_rbf: bool = True
    use_matern: bool = True
    matern_nu: float = 2.5
    use_poly: bool = True
    poly_degree: int = 2


@dataclass
class RetrievalSettings:
    # 候选召回与评测设置
    candidate_method: str = "bm25"  # bm25 / cosine
    rerank_top_n: int = 200
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    metrics_k: Tuple[int, int] = (10, 20)
    max_k: int = 20


@dataclass
class ContrastiveSettings:
    # 对比学习 baseline 参数
    projection_dim: int = 256
    temperature: float = 0.07
    epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 128


@dataclass
class ExperimentSettings:
    # 只跑指定实验，避免一股脑全部跑
    experiments: List[str] = field(default_factory=lambda: ["ours"])


@dataclass
class RunConfig:
    # 所有参数统一在这里修改，不使用命令行
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    gp: GPSettings = field(default_factory=GPSettings)
    kernel: KernelSettings = field(default_factory=KernelSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    contrastive: ContrastiveSettings = field(default_factory=ContrastiveSettings)
    experiments: ExperimentSettings = field(default_factory=ExperimentSettings)


def get_default_config() -> RunConfig:
    # 修改此函数中的默认值以调整实验参数
    config = RunConfig()
    # 允许用环境变量注入密钥，也可在此处写死
    config.embedding.api_base = os.getenv("EMBEDDING_API_BASE", config.embedding.api_base)
    config.embedding.api_key = os.getenv("EMBEDDING_API_KEY", config.embedding.api_key)
    return config
