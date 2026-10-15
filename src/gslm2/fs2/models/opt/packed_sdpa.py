# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scaled dot-product attention that dispatches packed batches to FlashAttention-2."""

from __future__ import annotations

from typing import final

from fairseq2.models.transformer.attention_bias import AttentionBias, AttentionBiasCache  # noqa: TC002
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.models.transformer.sdpa.torch import TorchSDPA
from fairseq2.nn import BatchLayout  # noqa: TC002
from torch import Tensor  # noqa: TC002

try:
    # fairseq2's flash2 module imports even without flash_attn and defers the failure to
    # call time, so probe flash_attn itself.
    import flash_attn  # noqa: F401
    from fairseq2.models.transformer.sdpa.flash2 import Flash2SDPA

    _HAS_FLASH2 = True
except ImportError:
    Flash2SDPA = None  # type: ignore[assignment,misc]
    _HAS_FLASH2 = False


@final
class PackedAwareSDPA(SDPA):
    """Routes packed batches to :class:`Flash2SDPA` and everything else to :class:`TorchSDPA`.

    Flash2SDPA takes the varlen path on packed layouts, avoiding a dense L*L mask, but
    rejects padded batches; TorchSDPA handles both. Falls back to TorchSDPA entirely when
    ``flash_attn`` is unavailable.
    """

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        super().__init__()
        self._torch = TorchSDPA(bias, dropout_p=dropout_p)
        self._flash = Flash2SDPA(bias, dropout_p=dropout_p) if _HAS_FLASH2 else None

    def forward(  # noqa: PLR0913
        self,
        q: Tensor,
        q_layout: BatchLayout,
        k: Tensor,
        k_layout: BatchLayout,
        v: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        # Flash2SDPA is CUDA-only and supports neither padded layouts nor attention weights.
        if self._flash is not None and q.is_cuda and q_layout.packed and not needs_weights:
            return self._flash(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)
        return self._torch(q, q_layout, k, k_layout, v, bias_cache, needs_weights=needs_weights)


def create_packed_aware_sdpa(bias: AttentionBias, *, dropout_p: float = 0.0) -> SDPA:
    """Return a :class:`PackedAwareSDPA` for ``bias``, mirroring ``create_default_sdpa``."""
    return PackedAwareSDPA(bias, dropout_p=dropout_p)
