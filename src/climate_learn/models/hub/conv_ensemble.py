# Local application
from .utils import register

# Third party
import torch
from torch import nn


@register("conv_ensemble")
class ConvEnsemble(nn.Module):
    """Улучшенная версия с несколькими слоями"""
    def __init__(
            self,
            n_models: int = 4,
            n_channels: int = 4,
            hidden_channels: int = 16
            ):
        super().__init__()
        
        self.network = nn.Sequential(
            # Первый слой: объединение моделей
            nn.Conv2d(n_models * n_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            
            # Второй слой: дополнительная обработка
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            
            # Третий слой: дополнительная обработка
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            
            # Финальный слой: проекция к нужному числу каналов
            nn.Conv2d(hidden_channels, n_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        height, width = x.shape[-2], x.shape[-1]
        x_reshaped = x.view(batch_size, -1, height, width)
        return self.network(x_reshaped)
