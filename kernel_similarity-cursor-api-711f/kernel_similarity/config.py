import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    corpus_path: str = "data/scifact/corpus.jsonl"
    queries_path: str = "data/scifact/queries.jsonl"
    qrels_path: str = "data/scifact/qrels/test.tsv"
    output_dir: str = "runs"
    split_path: str = "runs/query_split.json"



@dataclass
class EmbeddingSettings:
    # 嵌入接口配置（OpenAI 协议兼容）
    api_base: str = "https://yunwu.ai"
    api_key: str = "sk-"
    doc_model: str = "text-embedding-3-large"
    query_model: str = "text-embedding-3-large"
    batch_size: int = 32
    concurrency: int = 4
    timeout: float = 60.0
    cache_dir: str = "cache"
    cache_only: bool = True


@dataclass
class TrainingSettings:
    # 训练/采样控制参数
    seed: int = 42
    train_ratio: float = 0.9
    s_neg: int = 3
    s_pos: int = 0
    device: str = "cpu"


@dataclass
class GPSettings:
    # 稀疏 GP 训练参数（共享多核）
    inducing_points: int = 64
    inducing_init: str = "random"  # random / kmeans
    batch_queries: int = 4
    epochs: int = 0
    learning_rate: float = 1e-2
    use_inducing_points: bool = False


@dataclass
class KernelSettings:
    # 多核配置
    rbf_kernels: int = 0
    matern_nus: List[float] = field(default_factory=lambda: [])#2.5
    poly_degrees: List[int] = field(default_factory=lambda: [1])
    poly_offsets: List[float] = field(default_factory=lambda: [0.0])


@dataclass
class RetrievalSettings:
    # 候选召回与评测设置
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
    experiments: List[str] = field(default_factory=lambda: ["bm25", "cosine", "ours"])


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
