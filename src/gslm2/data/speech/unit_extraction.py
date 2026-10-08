"""On-the-fly audio to discrete-unit extraction: frozen-encoder codebook argmax, dedup of consecutive repeats over valid frames, shift past special tokens, chunk to the LM context window."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor
    from torch.nn import Module

    from gslm2.config import SpeechUnitConfig

# Minimum tokens the auto-regressive loss needs (input + target).
MIN_TOKENS_FOR_AR = 2
# Sample-unit cache files are bucketed into subdirs of this many indices to keep any one dir small.
_CACHE_SHARD_SIZE = 10_000


def audio_to_frame_lens(audio_lens: Tensor, conv_stack: Sequence[tuple[int, int]]) -> Tensor:
    """Frame count per sample after the feature-extractor conv stack.

    Applies ``floor((L - kernel) / stride) + 1`` per conv layer, matching the
    encoder's downsampling exactly (ceil(L/total_stride) over-estimates by ~1).
    """
    lens = audio_lens.to(dtype=torch.long)
    for kernel, stride in conv_stack:
        lens = torch.div(lens - kernel, stride, rounding_mode="floor") + 1
        lens = lens.clamp(min=0)
    return lens


def encoder_conv_stack(encoder: Module) -> tuple[tuple[int, int], ...]:
    """(kernel, stride) pairs of the encoder's feature-extractor conv stack."""
    return tuple((kernel, stride) for (_channels, kernel, stride) in encoder.cfg.extractor_conv_layer_config)


def per_file_units(codebook_logits: Tensor, frame_lens: Tensor, cfg: SpeechUnitConfig) -> list[Tensor]:
    """Per-file 1-D unit sequences: argmax, dedup over ``frame_lens`` valid frames,
    ``+cfg.unit_offset`` (reserving PAD=0/EOS=1), pad-to-min. No chunking — this is the
    cacheable per-file representation (fairseq2's ``pre_chunked_tokens``)."""
    device = codebook_logits.device
    discrete_tokens = codebook_logits.argmax(dim=-1)  # [B, T]
    batch_size, seq_len = discrete_tokens.shape

    valid_lens = frame_lens.to(device=device, dtype=torch.long).clamp(max=seq_len)
    valid_frames = torch.arange(seq_len, device=device).unsqueeze(0) < valid_lens.unsqueeze(1)
    if cfg.deduplicate and seq_len > 1:
        diff = discrete_tokens[:, 1:] != discrete_tokens[:, :-1]
        first = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        keep_mask = torch.cat([first, diff], dim=1) & valid_frames
    else:
        keep_mask = valid_frames

    out: list[Tensor] = []
    for i in range(batch_size):
        tokens = discrete_tokens[i][keep_mask[i]]
        if tokens.numel() == 0:
            tokens = torch.tensor([cfg.unit_offset], dtype=discrete_tokens.dtype, device=device)
        else:
            tokens = tokens + cfg.unit_offset
        if tokens.numel() < MIN_TOKENS_FOR_AR:
            tokens = torch.cat([tokens, tokens[-1:]])
        out.append(tokens)
    return out


def chunk_and_pad(per_file: list[Tensor], max_token_seq_len: int) -> tuple[Tensor, Tensor]:
    """Chunk per-file unit sequences to ``max_token_seq_len`` and right-pad the batch (PAD=0)."""
    processed: list[Tensor] = []
    seq_lens: list[int] = []
    for tokens in per_file:
        if tokens.numel() > max_token_seq_len:
            for chunk in tokens.split(max_token_seq_len):
                if chunk.numel() >= MIN_TOKENS_FOR_AR:
                    processed.append(chunk)
                    seq_lens.append(chunk.numel())
        else:
            processed.append(tokens)
            seq_lens.append(tokens.numel())
    device = processed[0].device if processed else torch.device("cpu")
    padded = pad_sequence(processed, batch_first=True, padding_value=0).long()
    return padded, torch.tensor(seq_lens, device=device)


def units_from_logits(codebook_logits: Tensor, frame_lens: Tensor, cfg: SpeechUnitConfig) -> tuple[Tensor, Tensor]:
    """Codebook logits ``[B, T, K]`` -> padded unit-token batch (dedup, offset, chunk, pad)."""
    return chunk_and_pad(per_file_units(codebook_logits, frame_lens, cfg), cfg.max_token_seq_len)


def _valid_frame_lens(attention_mask: Tensor | None, batch_size: int, num_frames: int, device: torch.device) -> Tensor:
    """Per-sample valid frame count from a ``[B, 1, T, T]`` (True=valid) mask."""
    if attention_mask is None:
        return torch.full((batch_size,), num_frames, dtype=torch.long, device=device)
    return attention_mask[:, 0, 0, :].sum(dim=-1).to(device=device, dtype=torch.long)


def _collect_codebook_logits(
    encoder: Module,
    waveforms: Tensor,
    attention_mask: Tensor | None,
    extract_layer: int,
    encoder_micro_batch: int,
) -> Tensor:
    """Run ``encoder.get_codebooks`` (micro-batched for large batches) -> ``[B, T, K]``."""

    def _one(wf: Tensor, mask: Tensor | None) -> Tensor:
        preds = encoder.get_codebooks(wf, attention_mask=mask, layers=[extract_layer], onehot=False)
        cb = next(c for c in reversed(preds) if c is not None)
        return cb.unsqueeze(0) if cb.dim() == 2 else cb  # noqa: PLR2004

    batch_size = waveforms.shape[0]
    if batch_size <= encoder_micro_batch:
        return _one(waveforms, attention_mask)

    codebooks: list[Tensor] = []
    for start in range(0, batch_size, encoder_micro_batch):
        end = start + encoder_micro_batch
        mask = attention_mask[start:end] if attention_mask is not None else None
        codebooks.append(_one(waveforms[start:end], mask))
    max_seq = max(cb.shape[1] for cb in codebooks)
    padded = [
        cb
        if cb.shape[1] == max_seq
        else torch.cat(
            [cb, torch.zeros(cb.shape[0], max_seq - cb.shape[1], cb.shape[2], device=cb.device, dtype=cb.dtype)],
            dim=1,
        )
        for cb in codebooks
    ]
    return torch.cat(padded, dim=0)


def extract_per_file_units(
    encoder: Module,
    waveforms: Tensor,
    attention_mask: Tensor | None,
    cfg: SpeechUnitConfig,
) -> list[Tensor]:
    """Encode raw audio to per-file 1-D unit sequences (frozen encoder; before chunking).

    ``attention_mask`` is the ``[B, 1, T_frame, T_frame]`` (True=valid) mask the speech
    collator emits. Returns one 1-D token tensor per input file — the cacheable form.
    """
    encoder_dtype = next(encoder.parameters()).dtype
    with torch.inference_mode():
        logits = _collect_codebook_logits(
            encoder,
            waveforms.to(dtype=encoder_dtype),
            attention_mask,
            cfg.extract_layer,
            cfg.encoder_micro_batch,
        )
    # inference_mode tensors can't flow into autograd bookkeeping downstream; clone out.
    logits = logits.clone()
    frame_lens = _valid_frame_lens(attention_mask, logits.shape[0], logits.shape[1], logits.device)
    return per_file_units(logits, frame_lens, cfg)


def extract_units(
    encoder: Module,
    waveforms: Tensor,
    attention_mask: Tensor | None,
    cfg: SpeechUnitConfig,
) -> tuple[Tensor, Tensor]:
    """Encode raw audio to a padded unit-token batch. ``(tokens [B', L], seq_lens [B'])``;
    ``B'`` may exceed the input batch when long sequences are chunked."""
    return chunk_and_pad(extract_per_file_units(encoder, waveforms, attention_mask, cfg), cfg.max_token_seq_len)


def index_cache_path(cache_dir: Path, index: int) -> Path:
    """Sharded on-disk path for a sample's cached per-file units."""
    return cache_dir / f"{index // _CACHE_SHARD_SIZE:05d}" / f"{index}.pt"


def path_cache_path(cache_dir: Path, fingerprint: str, audio_path: str) -> Path:
    """Stable on-disk path for one audio file's cached units.

    Unlike :func:`index_cache_path` (positional in one manifest), the key is a
    hash of the audio path, so entries stay valid across dev/pair manifests
    and repeated validations. ``fingerprint`` namespaces one encoder
    configuration; hex digests never collide with the trainer's numeric
    ``{index}.pt`` entries, so both layouts coexist in one cache dir with the
    same int32 per-file format.
    """
    import hashlib

    digest = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
    return cache_dir / fingerprint / f"{digest}.pt"


def save_units(path: Path, units: Tensor) -> None:
    """Atomically persist a 1-D unit tensor (temp file + rename) so a concurrent reader on
    another rank never observes a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(units.detach().to(device="cpu", dtype=torch.int32), tmp)
    tmp.replace(path)


def load_units(path: Path, device: torch.device) -> Tensor:
    """Load cached per-file units (written by :func:`save_units`) onto ``device`` as long."""
    return torch.load(path, map_location=device, weights_only=True).long()
