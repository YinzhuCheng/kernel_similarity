from dataclasses import dataclass
from typing import Dict, List, Tuple

import gpytorch
import numpy as np
import torch

from .kernels import WeightedKernel


@dataclass
class GPTrainingConfig:
    inducing_points: int
    inducing_init: str
    batch_queries: int
    epochs: int
    learning_rate: float
    device: str


class SparseGPClassifier(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, kernel: WeightedKernel):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _init_inducing_points(
    x: torch.Tensor, m: int, method: str, seed: int = 0
) -> torch.Tensor:
    if m >= x.size(0):
        return x.clone()
    if method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=m, random_state=seed, n_init=10)
        km.fit(x.cpu().numpy())
        return torch.tensor(km.cluster_centers_, device=x.device, dtype=x.dtype)
    rng = np.random.RandomState(seed)
    indices = rng.choice(x.size(0), size=m, replace=False)
    return x[indices].clone()


def build_query_models(
    train_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    kernel: WeightedKernel,
    config: GPTrainingConfig,
    seed: int,
) -> Dict[str, SparseGPClassifier]:
    models: Dict[str, SparseGPClassifier] = {}
    for idx, (qid, (x, _)) in enumerate(train_data.items()):
        inducing = _init_inducing_points(
            x, config.inducing_points, config.inducing_init, seed + idx
        )
        models[qid] = SparseGPClassifier(inducing, kernel).to(config.device)
    return models


def _unique_parameters(modules: List[torch.nn.Module]) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def train_multiquery_gp(
    train_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    models: Dict[str, SparseGPClassifier],
    kernel: WeightedKernel,
    config: GPTrainingConfig,
) -> Dict[str, float]:
    # 共享多核参数，在所有训练 query 上联合优化
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(config.device)
    mlls = {
        qid: gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=train_data[qid][0].size(0)
        )
        for qid, model in models.items()
    }
    optimizer = torch.optim.Adam(
        _unique_parameters([kernel, *models.values()]), lr=config.learning_rate
    )
    losses: Dict[str, float] = {}
    qids = list(train_data.keys())
    for epoch in range(config.epochs):
        kernel.train()
        np.random.shuffle(qids)
        epoch_loss = 0.0
        for start in range(0, len(qids), config.batch_queries):
            batch_qids = qids[start : start + config.batch_queries]
            optimizer.zero_grad()
            batch_loss = 0.0
            for qid in batch_qids:
                model = models[qid]
                model.train()
                x, y = train_data[qid]
                output = model(x)
                loss = -mlls[qid](output, y)
                batch_loss = batch_loss + loss
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        losses[f"epoch_{epoch+1}"] = epoch_loss
        print(f"epoch={epoch+1} loss={epoch_loss:.4f}")
    return losses


def train_single_query_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: WeightedKernel,
    config: GPTrainingConfig,
    seed: int,
) -> SparseGPClassifier:
    # 固定多核参数，仅训练单 query 的 GP 近似参数
    model = SparseGPClassifier(
        _init_inducing_points(x, config.inducing_points, config.inducing_init, seed),
        kernel,
    ).to(config.device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(config.device)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=x.size(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        model.train()
        kernel.train()
        output = model(x)
        loss = -mll(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def score_with_gp(
    model: SparseGPClassifier, x: torch.Tensor
) -> torch.Tensor:
    model.eval()
    output = model(x)
    return output.mean
