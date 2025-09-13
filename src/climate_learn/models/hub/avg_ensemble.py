# Local application
from .utils import register

# Third party
import torch
from torch import nn


@register("conv_ensemble")
class AvgEnsemble(nn.Module):
    """Average output over several models"""
    def __init__(
            self,
            ):
        super().__init__()
        # Add dummy parameter (used for fromal nn.Module assigning with 0 coefficient)
        self.dummy_param = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2) * (self.dummy_param * 0 + 1)
