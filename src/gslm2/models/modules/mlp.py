"""MLP / feed-forward modules for transformer blocks.

- ``MLP``: standard 2-layer FF with GELU + dropout (timm-style).
- ``SwiGLUFFN``: 3-projection SwiGLU FF used by LLaMA / VJEPA v3. No dropout
  (kept for interface parity only).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["MLP", "SwiGLUFFN"]


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


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward layer (LLaMA / VJEPA v3 style)."""

    def __init__(  # noqa: PLR0913
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type = nn.SiLU,
        drop: float = 0.0,
        *,
        wide_silu: bool = True,
    ) -> None:
        super().__init__()
        del act_layer, drop  # SwiGLU has no dropout / activation override; kept for parity
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        swiglu_hidden = hidden_features
        if wide_silu:
            swiglu_hidden = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden = (swiglu_hidden + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden)
        self.fc2 = nn.Linear(in_features, swiglu_hidden)
        self.fc3 = nn.Linear(swiglu_hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))
