"""Fairseq-free sLM21 minimal-pair scoring: rank each pair by (pseudo-)log-likelihood under the trained OPT over discrete units from the frozen speech encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence

from gslm2.data.common.audio_io import load_audio
from gslm2.data.common.masking.speech import build_attention_mask
from gslm2.data.speech.collators import conv_length
from gslm2.data.speech.unit_extraction import extract_per_file_units
from gslm2.models.core.scoring import multistream_sequence_log_likelihood, sequence_log_likelihood

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor
    from torch.nn import Module

    from gslm2.config import SpeechUnitConfig

logger = logging.getLogger(__name__)


@dataclass
class NativeSlmEvaluator:
    """Fairseq-free 2-alternative-forced-choice sLM21 scorer: a frozen speech encoder for unit
    extraction + a trained OPT for pseudo-log-likelihood scoring."""

    model: Module
    encoder: Module
    fe: SpeechUnitConfig
    device: torch.device
    batch_size: int = 32
    normalize: bool = True
    units_cache: dict[str, list[int]] | None = None

    def _units_for_paths(self, paths: list[str]) -> list[Tensor]:
        """Encode each audio file to its 1-D unit sequence (dedup + offset), capped at the LM context."""
        conv_layer_config = self.encoder.cfg.extractor_conv_layer_config
        units: list[Tensor] = []
        for start in range(0, len(paths), self.batch_size):
            waveforms = [load_audio(path, normalize=self.normalize) for path in paths[start : start + self.batch_size]]
            lengths = torch.tensor([wav.size(0) for wav in waveforms])
            padded = pad_sequence(waveforms, batch_first=True).to(self.device)
            causal_fe = getattr(self.encoder.cfg, "causal_fe", False)
            mask = build_attention_mask(conv_length(conv_layer_config, lengths, causal=causal_fe))
            attention_mask = mask.to(self.device) if mask is not None else None
            per_file = extract_per_file_units(self.encoder, padded, attention_mask, self.fe)
            units.extend(seq[: self.fe.max_token_seq_len] for seq in per_file)
        return units

    def cached_units_for_paths(self, paths: list[str]) -> list[Tensor]:
        """Per-path 1-D unit sequences, served from ``units_cache`` when attached."""
        if self.units_cache is None:
            return self._units_for_paths(paths)
        missing = [p for p in dict.fromkeys(paths) if p not in self.units_cache]
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            for path, seq in zip(chunk, self._units_for_paths(chunk), strict=True):
                self.units_cache[path] = seq.tolist()
        return [torch.tensor(self.units_cache[p], dtype=torch.long) for p in paths]

    def evaluate_benchmark(self, pair_manifest: str | Path) -> dict[str, float]:
        """2AFC accuracy on one benchmark's pair manifest: a pair is correct when
        ``LL(correct) > LL(incorrect)``. Returns raw-sum + length-normalized accuracy and count."""
        pairs = pl.read_csv(pair_manifest)
        correct = self.cached_units_for_paths(pairs["path_correct"].to_list())
        incorrect = self.cached_units_for_paths(pairs["path_incorrect"].to_list())

        n_correct = n_correct_norm = 0
        total = len(correct)
        for start in range(0, total, self.batch_size):
            ll_c, lln_c = sequence_log_likelihood(self.model, correct[start : start + self.batch_size], self.device)
            ll_i, lln_i = sequence_log_likelihood(self.model, incorrect[start : start + self.batch_size], self.device)
            n_correct += int((ll_c > ll_i).sum())
            n_correct_norm += int((lln_c > lln_i).sum())
        logger.info(
            "%s: acc=%.4f acc_norm=%.4f (n=%d)", pair_manifest, n_correct / total, n_correct_norm / total, total
        )
        return {"accuracy": n_correct / total, "accuracy_norm": n_correct_norm / total, "npairs": float(total)}


def _parse_multistream_units(units: str) -> Tensor:
    """Parse a ``|``-joined multi-stream unit string (``"u u u|v v v|..."``) into an ``[S, L]`` tensor,
    truncating the streams to their common (min) length so they stay aligned."""
    streams = [[int(token) for token in part.split()] for part in units.split("|")]
    length = min(len(stream) for stream in streams)
    return torch.tensor([stream[:length] for stream in streams], dtype=torch.long)


@dataclass
class NativeMultiStreamSlmEvaluator:
    """Fairseq-free 2AFC sLM21 scorer for a multi-stream model over pre-extracted ``|``-joined unit pairs.

    Unlike :class:`NativeSlmEvaluator` (which encodes audio on the fly), multi-stream benchmarks carry
    pre-extracted, aligned per-stream codes — matching the fairseq2 multi-stream eval — so the manifest
    provides ``positive`` / ``negative`` ``|``-joined unit strings rather than audio paths.
    """

    model: Module
    device: torch.device
    batch_size: int = 32
    dedup_aware: bool = True

    def evaluate_benchmark(self, pair_manifest: str | Path) -> dict[str, float]:
        """2AFC accuracy on one benchmark's ``|``-joined unit-pair manifest (jsonl or csv with
        ``positive`` / ``negative`` columns). A pair is correct when ``LL(positive) > LL(negative)``."""
        path = str(pair_manifest)
        pairs = pl.read_ndjson(path) if path.endswith((".jsonl", ".ndjson")) else pl.read_csv(path)
        positive = [_parse_multistream_units(u) for u in pairs["positive"].to_list()]
        negative = [_parse_multistream_units(u) for u in pairs["negative"].to_list()]

        n_correct = n_correct_norm = 0
        total = len(positive)
        for start in range(0, total, self.batch_size):
            stop = start + self.batch_size
            ll_c, lln_c = multistream_sequence_log_likelihood(
                self.model, positive[start:stop], self.device, dedup_aware=self.dedup_aware
            )
            ll_i, lln_i = multistream_sequence_log_likelihood(
                self.model, negative[start:stop], self.device, dedup_aware=self.dedup_aware
            )
            n_correct += int((ll_c > ll_i).sum())
            n_correct_norm += int((lln_c > lln_i).sum())
        logger.info(
            "%s: acc=%.4f acc_norm=%.4f (n=%d)", pair_manifest, n_correct / total, n_correct_norm / total, total
        )
        return {"accuracy": n_correct / total, "accuracy_norm": n_correct_norm / total, "npairs": float(total)}
