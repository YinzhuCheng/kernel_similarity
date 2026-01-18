from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class ContrastiveConfig:
    projection_dim: int
    temperature: float
    epochs: int
    learning_rate: float
    batch_size: int
    device: str


class ContrastiveRetriever(torch.nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, temperature: float):
        super().__init__()
        self.query_proj = torch.nn.Linear(input_dim, projection_dim, bias=False)
        self.doc_proj = torch.nn.Linear(input_dim, projection_dim, bias=False)
        self.temperature = temperature

    def forward(self, query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query)
        d = self.doc_proj(doc)
        q = torch.nn.functional.normalize(q, dim=-1)
        d = torch.nn.functional.normalize(d, dim=-1)
        return (q * d).sum(dim=-1) / self.temperature


def train_contrastive(
    pairs: List[Tuple[np.ndarray, np.ndarray, int]],
    input_dim: int,
    config: ContrastiveConfig,
) -> ContrastiveRetriever:
    model = ContrastiveRetriever(input_dim, config.projection_dim, config.temperature).to(
        config.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    if not pairs:
        return model
    for epoch in range(config.epochs):
        np.random.shuffle(pairs)
        epoch_loss = 0.0
        for start in range(0, len(pairs), config.batch_size):
            batch = pairs[start : start + config.batch_size]
            q = torch.tensor(
                np.stack([row[0] for row in batch]),
                dtype=torch.float32,
                device=config.device,
            )
            d = torch.tensor(
                np.stack([row[1] for row in batch]),
                dtype=torch.float32,
                device=config.device,
            )
            y = torch.tensor(
                [row[2] for row in batch],
                dtype=torch.float32,
                device=config.device,
            )
            optimizer.zero_grad()
            logits = model(q, d)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"contrastive epoch={epoch+1} loss={epoch_loss:.4f}")
    return model


@torch.no_grad()
def score_contrastive(
    model: ContrastiveRetriever,
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
    device: str,
) -> np.ndarray:
    model.eval()
    q = torch.tensor(query_vec, dtype=torch.float32, device=device).unsqueeze(0)
    d = torch.tensor(doc_matrix, dtype=torch.float32, device=device)
    scores = model(q.repeat(d.size(0), 1), d)
    return scores.cpu().numpy()
