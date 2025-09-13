# Local application
from .utils import register

# Third party
import torch
from torch import nn


@register("conv_ensemble")
class AvgEnsemble(nn.Module):
    """Улучшенная версия с несколькими слоями"""
    def __init__(
            self,
            ):
        super().__init__()
        # Добавляем dummy-параметр (обучаемый, но не используемый в вычислениях)
        self.dummy_param = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2) * (self.dummy_param * 0 + 1)
