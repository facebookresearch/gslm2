"""Pre-extracted multi-stream speech-unit dataset for ``MultiStreamOPT`` training.

Each manifest ``.txt`` line holds the ``num_streams`` aligned streams concatenated as equal-length
space-separated blocks (stream 0's tokens, then stream 1's, ...). A line is reshaped to ``[S, L]`` and the
collator batches these into ``[B, S, T]`` (right-padded with ``pad_idx``) with next-token targets per
stream. Mirrors the fairseq2 multi-stream unit dataset loader (raw codebook ids
offset by ``unit_offset`` at load; the padded/last-position targets are ``pad_idx``, the LM's ignore index).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from gslm2.data.common.weighted_sampling import WeightedDistributedBatchSampler

if TYPE_CHECKING:
    from torch import Tensor

    from gslm2.config.data import PreExtractedUnitSource


def _read_unit_lines(manifest_path: str) -> list[list[int]]:
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        return [[int(token) for token in line.split()] for line in f if line.strip()]


class MultiStreamUnitDataset(Dataset):
    """Reads pre-extracted multi-stream unit lines (concatenated per-stream blocks) from ``.txt`` manifests."""

    def __init__(self, source: PreExtractedUnitSource, num_streams: int) -> None:
        super().__init__()
        self.num_streams = num_streams
        self.lines: list[Tensor] = []
        self.dataset_sizes: list[int] = []
        for manifest_path in source.manifests:
            kept = 0
            for tokens in _read_unit_lines(manifest_path):
                stream_len = len(tokens) // self.num_streams
                if stream_len < source.min_seq_len or stream_len > source.max_seq_len:
                    continue
                usable = self.num_streams * stream_len  # drop a ragged tail so the blocks stay aligned
                self.lines.append(torch.tensor(tokens[:usable], dtype=torch.long) + source.unit_offset)
                kept += 1
            self.dataset_sizes.append(kept)
        if not self.lines:
            raise ValueError(f"No usable multi-stream unit lines in manifests: {source.manifests}")

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> Tensor:
        return self.lines[index]


@dataclass
class MultiStreamUnitCollator:
    """Batches concatenated-block unit lines into ``(streams, targets, attention_mask)``.

    ``streams`` ``[B, S, T]`` (pad=``pad_idx``); ``targets`` next-token per stream (``pad_idx`` at pad and the
    final position — the LM's ignore index); ``attention_mask`` ``[B, T]`` (1=valid). Each input line is a
    1-D tensor of ``S*L`` tokens (``S`` equal blocks) reshaped to ``[S, L]``.
    """

    num_streams: int
    pad_idx: int = 1

    def __call__(self, samples: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        per_sample = [sample.reshape(self.num_streams, -1) for sample in samples]  # [S, L_i]
        lengths = [x.size(1) for x in per_sample]
        max_len = max(lengths)
        streams = torch.full((len(samples), self.num_streams, max_len), self.pad_idx, dtype=torch.long)
        attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
        for i, (block, length) in enumerate(zip(per_sample, lengths, strict=True)):
            streams[i, :, :length] = block
            attention_mask[i, :length] = 1
        # Next-token targets per stream; padded streams are already pad_idx, so the shifted pads and the
        # last column stay pad_idx (which MultiStreamOPT ignores in its cross-entropy).
        targets = torch.full_like(streams, self.pad_idx)
        targets[:, :, :-1] = streams[:, :, 1:]
        return streams, targets, attention_mask


@dataclass
class MultiStreamUnitPackedCollator:
    """Packs several documents per row: ``(streams [B,S,T], targets [B,S,T], attention_mask [B,T], doc_ids [B,T])``.

    Greedy bin-packing to ``<= max_len`` frames per row removes padding waste; ``doc_ids`` give per-token
    document membership (``-1`` padding) for the model's block-diagonal attention, and ``targets`` are the
    within-document next token (a document's last frame and padding stay ``pad_idx``, the LM's ignore index).
    Mirrors the fairseq2 multi-stream packing; ``MultiStreamOPT`` is bit-exact per document vs the padded path.
    """

    num_streams: int
    max_len: int
    pad_idx: int = 1

    def __call__(self, samples: list[Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        docs = [sample.reshape(self.num_streams, -1)[:, : self.max_len] for sample in samples]  # each [S, L]
        bins: list[list[Tensor]] = [[]]
        bin_lens = [0]
        for doc in docs:
            if bin_lens[-1] + doc.size(1) > self.max_len and bin_lens[-1] > 0:
                bins.append([])
                bin_lens.append(0)
            bins[-1].append(doc)
            bin_lens[-1] += doc.size(1)

        t_max = max(bin_lens)
        streams = torch.full((len(bins), self.num_streams, t_max), self.pad_idx, dtype=torch.long)
        targets = torch.full((len(bins), self.num_streams, t_max), self.pad_idx, dtype=torch.long)
        doc_ids = torch.full((len(bins), t_max), -1, dtype=torch.long)
        attention_mask = torch.zeros((len(bins), t_max), dtype=torch.long)
        for row, docs_in_bin in enumerate(bins):
            offset = 0
            for doc_index, doc in enumerate(docs_in_bin):
                length = doc.size(1)
                streams[row, :, offset : offset + length] = doc
                doc_ids[row, offset : offset + length] = doc_index
                attention_mask[row, offset : offset + length] = 1
                if length > 1:  # within-doc next token; the doc's last frame stays pad_idx (no cross-doc predict)
                    targets[row, :, offset : offset + length - 1] = doc[:, 1:]
                offset += length
        return streams, targets, attention_mask, doc_ids


def build_multistream_unit_dataloader(source: PreExtractedUnitSource, num_streams: int) -> DataLoader:
    """DataLoader over pre-extracted multi-stream unit manifests.

    Yields ``(streams, targets, attention_mask)`` when padded, or ``(streams, targets, attention_mask, doc_ids)``
    when ``source.pack`` is set.
    """
    dataset = MultiStreamUnitDataset(source, num_streams)
    collate_fn: MultiStreamUnitCollator | MultiStreamUnitPackedCollator = (
        MultiStreamUnitPackedCollator(num_streams=num_streams, max_len=source.max_seq_len, pad_idx=source.pad_idx)
        if source.pack
        else MultiStreamUnitCollator(num_streams=num_streams, pad_idx=source.pad_idx)
    )

    sample_weights: list[float] = []
    for weight, size in zip(source.manifests_weights, dataset.dataset_sizes, strict=True):
        sample_weights.extend([weight / size] * size)
    batch_sampler = WeightedDistributedBatchSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        batch_size=source.batch_size,
        seed=source.random_seed,
        drop_last=source.drop_last,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=source.num_workers,
        collate_fn=collate_fn,
        pin_memory=source.pin_memory,
        prefetch_factor=source.prefetch_factor if source.num_workers > 0 else None,
        persistent_workers=source.persistent_workers and source.num_workers > 0,
        generator=torch.Generator().manual_seed(
            source.random_seed + (dist.get_rank() if dist.is_initialized() else 0)
        ),
    )
