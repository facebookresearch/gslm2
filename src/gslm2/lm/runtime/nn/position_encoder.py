# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, final, override

import torch
from fairseq2.nn import PositionEncoder
from fairseq2.ops import unsqueeze
from torch import Tensor, nn
from torch.nn.functional import embedding
from torch.nn.parameter import Parameter

if TYPE_CHECKING:
    from fairseq2.data_type import DataType
    from fairseq2.device import Device
    from fairseq2.nn.batch_layout import BatchLayout
    from fairseq2.nn.incremental_state import IncrementalStateBag


@final
class LearnedPositionEncoder(PositionEncoder):
    """Encodes sequences with learned positional embeddings."""

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int,
        *,
        _legacy_pad_idx: int | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param encoding_dim: The dimensionality of positional encodings. The
            last dimension of input sequences is expected to have the same
            dimensionality.
        :param max_seq_len: The maximum allowed length for input sequences.
            Sequences longer than ``max_seq_len`` will cause a :class:`ValueError`.
        """
        super().__init__(encoding_dim)

        self.weight = Parameter(torch.empty((max_seq_len + 1, encoding_dim), device=device, dtype=dtype))

        self.max_seq_len = max_seq_len

        # This is a legacy parameter that should only be set when the encodings
        # must be compatible with fairseq.
        self._legacy_pad_idx = 0 if _legacy_pad_idx is None else _legacy_pad_idx

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

        with torch.no_grad():
            self.weight[0].fill_(0.0)  # pad

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        start_step = state_bag.step_nr if not self.training and state_bag is not None else 0

        max_seq_len = start_step + seqs_layout.max_seq_len

        if max_seq_len > self.max_seq_len:
            raise ValueError(
                f"The lengths of all sequences in `seqs` must be less than or equal to the maximum sequence length ({self.max_seq_len}), but at least one sequence is of length {max_seq_len} instead."
            )

        indices = seqs_layout.position_indices + (1 + self._legacy_pad_idx)

        if not self.training and state_bag is not None:
            indices = state_bag.step_nr + indices

        # ([N], S, E)
        embeds = embedding(indices, self.weight, padding_idx=0)

        if d := seqs.ndim - embeds.ndim:
            embeds = unsqueeze(embeds, dim=-2, count=d)

        return seqs + embeds

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"encoding_dim={self.encoding_dim}, max_seq_len={self.max_seq_len}"
