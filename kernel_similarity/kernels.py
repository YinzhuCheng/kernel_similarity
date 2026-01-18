from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gpytorch
import torch


@dataclass
class KernelConfig:
    rbf_kernels: int = 1
    matern_nus: List[float] = field(default_factory=lambda: [2.5])
    poly_degrees: List[int] = field(default_factory=lambda: [2])
    poly_offsets: List[float] = field(default_factory=lambda: [0.0])


class WeightedKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        base_kernels: List[gpytorch.kernels.Kernel],
        kernel_labels: Optional[List[str]] = None,
        initial_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        if not base_kernels:
            raise ValueError("At least one base kernel is required.")
        if kernel_labels and len(kernel_labels) != len(base_kernels):
            raise ValueError("Kernel labels size mismatch.")
        self.base_kernels = torch.nn.ModuleList(base_kernels)
        self.kernel_labels = kernel_labels or []
        self.alpha = torch.nn.Parameter(torch.zeros(len(base_kernels)))
        if initial_weights is not None:
            self._init_alpha(initial_weights)

    def forward(self, x1, x2, **params):
        weights = torch.softmax(self.alpha, dim=0)
        covar = None
        for weight, kernel in zip(weights, self.base_kernels):
            kernel_covar = kernel(x1, x2, **params)
            term = kernel_covar.mul(weight)
            covar = term if covar is None else covar + term
        return covar

    def kernel_summary(self) -> List[str]:
        weights = torch.softmax(self.alpha.detach(), dim=0)
        summaries = []
        for idx, (weight, kernel) in enumerate(zip(weights, self.base_kernels)):
            label = (
                self.kernel_labels[idx]
                if self.kernel_labels and idx < len(self.kernel_labels)
                else kernel.__class__.__name__
            )
            summaries.append(f"{label}:{weight.item():.4f}")
        return summaries

    def _init_alpha(self, initial_weights: List[float]) -> None:
        if len(initial_weights) != len(self.base_kernels):
            raise ValueError("Initial weights size mismatch.")
        weights = torch.tensor(initial_weights, dtype=self.alpha.dtype)
        if torch.any(weights < 0):
            raise ValueError("Initial weights must be non-negative.")
        total = weights.sum()
        if total <= 0:
            raise ValueError("Initial weights must sum to a positive value.")
        weights = weights / total
        logits = torch.log(torch.clamp(weights, min=1e-6))
        with torch.no_grad():
            self.alpha.copy_(logits)


def _normalize_poly_offsets(
    poly_offsets: List[float], poly_degrees: List[int]
) -> List[float]:
    if not poly_offsets:
        raise ValueError("poly_offsets cannot be empty.")
    if len(poly_offsets) == 1:
        return [poly_offsets[0] for _ in poly_degrees]
    if len(poly_offsets) != len(poly_degrees):
        raise ValueError("poly_offsets must have length 1 or match poly_degrees.")
    return poly_offsets


def _validate_matern_nus(matern_nus: List[float]) -> None:
    allowed = {0.5, 1.5, 2.5}
    invalid = [nu for nu in matern_nus if nu not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported Matern nu values: {invalid}. Allowed: {sorted(allowed)}."
        )


def build_base_kernels(
    config: KernelConfig,
) -> Tuple[List[gpytorch.kernels.Kernel], List[str], Optional[int]]:
    kernels: List[gpytorch.kernels.Kernel] = []
    labels: List[str] = []
    linear_poly_idx: Optional[int] = None
    if config.rbf_kernels < 0:
        raise ValueError("rbf_kernels must be non-negative.")
    rbf_kernels = config.rbf_kernels
    for idx in range(rbf_kernels):
        kernels.append(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        labels.append(f"RBF#{idx + 1}")
    matern_nus = config.matern_nus or []
    if matern_nus:
        _validate_matern_nus(matern_nus)
        for nu in matern_nus:
            kernels.append(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=nu)
                )
            )
            labels.append(f"Matern(nu={nu})")
    poly_degrees = config.poly_degrees or []
    if poly_degrees:
        poly_offsets = _normalize_poly_offsets(config.poly_offsets or [0.0], poly_degrees)
        for degree, offset in zip(poly_degrees, poly_offsets):
            if degree <= 0:
                raise ValueError("Polynomial degree must be positive.")
            kernels.append(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.PolynomialKernel(
                        power=degree, offset=offset
                    )
                )
            )
            labels.append(f"Poly(degree={degree}, offset={offset})")
            if degree == 1 and linear_poly_idx is None:
                linear_poly_idx = len(kernels) - 1
    return kernels, labels, linear_poly_idx


def build_weighted_kernel(config: KernelConfig) -> WeightedKernel:
    base_kernels, labels, linear_poly_idx = build_base_kernels(config)
    initial_weights = None
    if linear_poly_idx is not None:
        initial_weights = [0.0 for _ in base_kernels]
        initial_weights[linear_poly_idx] = 1.0
    return WeightedKernel(
        base_kernels, kernel_labels=labels, initial_weights=initial_weights
    )
