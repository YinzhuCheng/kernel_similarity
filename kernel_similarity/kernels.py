from dataclasses import dataclass
from typing import List

import gpytorch
import torch


@dataclass
class KernelConfig:
    use_rbf: bool = True
    use_matern: bool = True
    matern_nu: float = 2.5
    use_poly: bool = True
    poly_degree: int = 2


class WeightedKernel(gpytorch.kernels.Kernel):
    def __init__(self, base_kernels: List[gpytorch.kernels.Kernel]):
        super().__init__()
        if not base_kernels:
            raise ValueError("At least one base kernel is required.")
        self.base_kernels = torch.nn.ModuleList(base_kernels)
        self.alpha = torch.nn.Parameter(torch.zeros(len(base_kernels)))

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
        for weight, kernel in zip(weights, self.base_kernels):
            summaries.append(f"{kernel.__class__.__name__}:{weight.item():.4f}")
        return summaries


def build_base_kernels(config: KernelConfig) -> List[gpytorch.kernels.Kernel]:
    kernels: List[gpytorch.kernels.Kernel] = []
    if config.use_rbf:
        kernels.append(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
    if config.use_matern:
        kernels.append(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=config.matern_nu)
            )
        )
    if config.use_poly:
        kernels.append(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PolynomialKernel(power=config.poly_degree)
            )
        )
    return kernels


def build_weighted_kernel(config: KernelConfig) -> WeightedKernel:
    return WeightedKernel(build_base_kernels(config))
