"""Token-selection masking and multi-clip wrappers for JEPA models.

Two flavors of functionality:

1. **Token-selection masks** (``apply_masks``, ``repeat_interleave_batch``) —
   index tensors ``[B, K]`` that pick out a subset of tokens from a
   ``[B, N, D]`` sequence. Used by JEPA-style context/target splits.
2. **Multi-clip / multi-mask wrappers** (``EncoderWrapper`` /
   ``PredictorWrapper``) — thin modules that fan out an encoder / predictor
   backbone over nested lists of clips and per-clip mask configurations.
   Used by both VJEPA and AJEPA on top of their respective backbones.

"""

from __future__ import annotations

import torch
from torch import nn


def apply_masks(
    x: torch.Tensor,
    masks: list[torch.Tensor],
    concat: bool = True,  # noqa: FBT001, FBT002
) -> torch.Tensor | list[torch.Tensor]:
    """Gather tokens at mask indices.

    Args:
        x: ``(B, N, D)`` tensor of patch tokens.
        masks: list of ``(B, K)`` index tensors selecting K patches from N.
        concat: if True, concatenate along batch dim; else return list.

    Returns:
        ``(B*len(masks), K, D)`` tensor of selected tokens, or list of
        ``(B, K, D)`` tensors.
    """
    if not isinstance(masks, list):
        masks = [masks]
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x: torch.Tensor, b: int, repeat: int) -> torch.Tensor:
    """Repeat-interleave along the batch dimension.

    Splits ``x`` into chunks of size ``b`` and repeats each chunk ``repeat``
    times, concatenating along the batch dim.
    """
    n = len(x) // b
    return torch.cat(
        [torch.cat([x[i * b : (i + 1) * b] for _ in range(repeat)], dim=0) for i in range(n)],
        dim=0,
    )


class EncoderWrapper(nn.Module):
    """Fan an encoder backbone over a list of clips x list of mask groups.

    Backbone contract: ``backbone(x, masks=mask)`` where ``mask`` is a
    ``[B, K]`` index tensor (or ``None`` for the full sequence). Used by both
    VJEPA (``VisionTransformer``) and AJEPA (``AudioEncoder``).

    When the backbone returns a ``dict[int, Tensor]`` (hierarchical mode),
    the wrapper propagates it as-is into the output lists.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

    def forward(
        self,
        x: list[torch.Tensor] | torch.Tensor,
        masks: list[list[torch.Tensor]] | None = None,
        **kwargs: torch.Tensor,
    ) -> list:
        if not isinstance(x, list):
            x = [x]
        if masks is None:
            return [self.backbone(xi, **kwargs) for xi in x]
        outs: list[list[torch.Tensor]] = [[] for _ in x]
        for i, (xi, mi) in enumerate(zip(x, masks, strict=True)):
            for mij in mi:
                outs[i].append(self.backbone(xi, masks=mij, **kwargs))
        return outs


class PredictorWrapper(nn.Module):
    """Fan a predictor backbone over a list of clips x list of mask groups.

    Backbone contract: ``backbone(x, masks_x, masks_y, mask_index=j)`` where
    ``j`` is the per-clip mask-group index (selects the predictor's mask token).
    Used by both VJEPA (``VisionTransformerPredictor``) and AJEPA
    (``AudioPredictor``).

    When the backbone is a :class:`~gslm2.models.modules.hierarchical.HierarchicalPredictorWrapper`,
    ``x`` elements are ``dict[int, Tensor]`` (per-layer features) rather than
    plain tensors. The wrapper passes them through transparently.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        x: list,
        masks_x: list,
        masks_y: list,
        **kwargs: torch.Tensor,
    ) -> list:
        if not isinstance(x, list):
            x = [x]
        outs: list[list[torch.Tensor]] = [[] for _ in x]
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y, strict=True)):
            for j, (xij, mxij, myij) in enumerate(zip(xi, mxi, myi, strict=True)):
                outs[i].append(self.backbone(xij, mxij, myij, mask_index=j, **kwargs))
        return outs


__all__ = [
    "EncoderWrapper",
    "PredictorWrapper",
    "apply_masks",
    "repeat_interleave_batch",
]
