"""Feed-forward MLP for language model transformer blocks."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["MLP"]


class MLP(nn.Module):
    """Standard 2-layer feed-forward MLP."""

    def __init__(  # noqa: PLR0913 -- feed-forward constructor mirrors the timm Mlp signature
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
        *,
        post_drop: bool = True,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._post_drop = post_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self._post_drop:
            x = self.drop(x)
        return x
