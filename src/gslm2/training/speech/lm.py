"""``SLMTrainer``: speech-unit language-model pretraining.

On-the-fly audio->unit extraction with a frozen speech encoder, then causal LM over the unit
stream. (Pre-extracted units are just a token manifest and train via
:class:`gslm2.training.text.lm.CausalLMTrainer`.)
"""

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gslm2.config import (
    DinoSRConfig,
    LMConfig,
    OnTheFlyUnitSource,
    PreExtractedUnitSource,
    SLMUnitDataConfig,
    SpeechUnitConfig,
    SpidRConfig,
)
from gslm2.data.speech.dataloader import build_dataloader as build_speech_dataloader
from gslm2.data.speech.multistream import build_multistream_unit_dataloader
from gslm2.data.speech.unit_extraction import (
    chunk_and_pad,
    extract_per_file_units,
    index_cache_path,
    load_units,
    save_units,
)
from gslm2.data.text.collators import _IGNORE_INDEX
from gslm2.models.factory import build_model
from gslm2.training.base import BaseTrainer

logger = logging.getLogger()


SLM_RECORDS_EXTRA_KEYS: list[str] = ["perplexity", "nll_loss"]
# The two terms the multi-stream loss blends, averaged per logging interval alongside "loss".
MULTISTREAM_SLM_RECORDS_EXTRA_KEYS: list[str] = ["stream_specific_loss", "gen_loss"]

_ENCODER_CONFIGS = {"spidr": SpidRConfig, "dinosr": DinoSRConfig}


class SLMTrainer(BaseTrainer):
    """On-the-fly SLM trainer: frozen speech encoder -> unit extraction -> causal LM."""

    def __init__(self, cfg: LMConfig) -> None:
        super().__init__(
            cfg,
            records_extra_keys=SLM_RECORDS_EXTRA_KEYS,
            pbar_keys=["loss", "perplexity"],
        )
        self._encoder: torch.nn.Module | None = None
        # Shared on-disk cache of per-file units, keyed by dataset index.
        self._unit_cache_dir: Path | None = None

    def _build_components(self) -> None:
        # Load the frozen encoder before super() so build_dataloader can read its
        # conv-stack for the audio collator's frame mask.
        data = self.cfg.data
        assert isinstance(data, SLMUnitDataConfig)
        assert isinstance(data.source, OnTheFlyUnitSource)
        source = data.source
        self._encoder = self._load_encoder(source.feature_extraction)
        if source.feature_extraction.unit_cache_dir:
            self._unit_cache_dir = Path(source.feature_extraction.unit_cache_dir)
            self._unit_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("On-the-fly SLM unit cache: %s", self._unit_cache_dir)
            if source.audio.rand_crop:
                logger.warning("unit_cache_dir set with rand_crop=True: cached units freeze the first crop seen.")
        super()._build_components()

    def _load_encoder(self, fe: SpeechUnitConfig) -> torch.nn.Module:
        if fe.checkpoint:
            encoder = build_model(model_type=fe.model_type, checkpoint=fe.checkpoint)
        else:  # random encoder (smoke tests / throughput benchmark — weights don't affect speed)
            encoder = build_model(cfg=_ENCODER_CONFIGS[fe.model_type](), model_type=fe.model_type)
        encoder = encoder.to(self.device, dtype=getattr(torch, fe.dtype)).train(mode=False)
        encoder.requires_grad_(requires_grad=False)
        logger.info(
            "Loaded frozen %s encoder for on-the-fly SLM (checkpoint=%r)", fe.model_type, fe.checkpoint or None
        )
        return encoder

    def _feature_extraction(self) -> SpeechUnitConfig:
        """The on-the-fly source's frozen-encoder config (this trainer only runs on OnTheFlyUnitSource)."""
        data = self.cfg.data
        assert isinstance(data, SLMUnitDataConfig)
        assert isinstance(data.source, OnTheFlyUnitSource)
        return data.source.feature_extraction

    def build_dataloader(self, cfg: LMConfig) -> Iterable:  # type: ignore[override]
        data = cfg.data
        assert isinstance(data, SLMUnitDataConfig)
        assert isinstance(data.source, OnTheFlyUnitSource)
        assert self._encoder is not None
        return build_speech_dataloader(
            data.source.audio,
            mask_cfg=None,
            conv_layer_config=self._encoder.cfg.extractor_conv_layer_config,
            causal_fe=getattr(self._encoder.cfg, "causal_fe", False),
        )

    def training_step(self, batch: Any, step: int) -> tuple[Tensor, dict[str, Any]]:  # noqa: ANN401, ARG002
        fe = self._feature_extraction()
        waveforms, attention_mask, indices = batch
        per_file = self._extract_units(waveforms, attention_mask, indices.tolist())
        tokens, _ = chunk_and_pad(per_file, fe.max_token_seq_len)
        # Next-token causal targets over the unit sequence (matches CausalCollator; PAD=0).
        pad = tokens == 0
        targets = torch.full_like(tokens, _IGNORE_INDEX)
        targets[:, :-1] = tokens[:, 1:].masked_fill(pad[:, 1:], _IGNORE_INDEX)
        with torch.autocast(self.device.type, self.dtype, self.cfg.optimizer.mixed_precision):
            loss, log_dict = self.ddp_model(tokens, targets=targets, attention_mask=(~pad).long())
        metrics: dict[str, Any] = {"loss": loss.detach(), "batch_size": tokens.size(0)}
        metrics.update(log_dict)
        return loss, metrics

    def _extract_units(self, waveforms: Tensor, attention_mask: Tensor | None, idx_list: list[int]) -> list[Tensor]:
        """Per-file units for the batch, reading the on-disk cache when present and encoding
        (then caching) only the misses. Without a cache dir, encode the whole batch."""
        assert self._encoder is not None
        fe = self._feature_extraction()
        mask = attention_mask.to(self.device, non_blocking=True) if attention_mask is not None else None
        if self._unit_cache_dir is None:
            return extract_per_file_units(
                self._encoder,
                waveforms.to(self.device, non_blocking=True),
                mask,
                fe,
            )
        paths = [index_cache_path(self._unit_cache_dir, idx) for idx in idx_list]
        per_file: list[Tensor | None] = [None] * len(idx_list)
        to_encode = [pos for pos, path in enumerate(paths) if not path.exists()]
        for pos, path in enumerate(paths):
            if pos not in to_encode:
                per_file[pos] = load_units(path, self.device)
        if to_encode:
            sel = torch.tensor(to_encode)
            encoded = extract_per_file_units(
                self._encoder,
                waveforms[sel].to(self.device, non_blocking=True),
                mask[sel] if mask is not None else None,
                fe,
            )
            for pos, units in zip(to_encode, encoded, strict=True):
                per_file[pos] = units
                save_units(paths[pos], units)
        return [units for units in per_file if units is not None]

    def _extra_banner_fields(self) -> dict[str, object]:
        fe = self._feature_extraction()
        return {"Model type": self.cfg.run.model_type, "Task": f"on-the-fly SLM ({fe.model_type})"}


def train_slm(cfg: LMConfig) -> None:
    SLMTrainer(cfg).fit()


class MultiStreamSLMTrainer(BaseTrainer):
    """Multi-stream SLM trainer: ``MultiStreamOPT`` over pre-extracted multi-stream unit manifests.

    Units are pre-extracted (no encoder at train time), so this reads ``[B, S, T]`` batches directly and
    lets the model's per-stream + depformer loss pick the stream-specific stream internally.
    """

    def __init__(self, cfg: LMConfig) -> None:
        super().__init__(cfg, records_extra_keys=MULTISTREAM_SLM_RECORDS_EXTRA_KEYS, pbar_keys=["loss"])

    def build_dataloader(self, cfg: LMConfig) -> Iterable:  # type: ignore[override]
        data = cfg.data
        assert isinstance(data, SLMUnitDataConfig)
        assert isinstance(data.source, PreExtractedUnitSource)
        return build_multistream_unit_dataloader(data.source, data.num_streams)

    def training_step(self, batch: Any, step: int) -> tuple[Tensor, dict[str, Any]]:  # noqa: ANN401, ARG002
        tensors = [t.to(self.device, non_blocking=True) for t in batch]
        streams, targets, attention_mask = tensors[0], tensors[1], tensors[2]
        doc_ids = tensors[3] if len(tensors) == 4 else None  # packed collator adds doc_ids  # noqa: PLR2004
        # Draw the stream-specific stream here (a Python int) rather than inside the model, so the scalar
        # sync doesn't force a torch.compile graph break in the hot path.
        stream_idx = int(torch.randint(self.cfg.model.num_streams, (1,)).item())
        with torch.autocast(self.device.type, self.dtype, self.cfg.optimizer.mixed_precision):
            loss, log_dict = self.ddp_model(
                streams, targets=targets, attention_mask=attention_mask, doc_ids=doc_ids, stream_idx=stream_idx
            )
        metrics: dict[str, Any] = {"loss": loss.detach(), "batch_size": streams.size(0)}
        metrics.update(log_dict)
        return loss, metrics

    def _extra_banner_fields(self) -> dict[str, object]:
        num_streams = self.cfg.data.num_streams if isinstance(self.cfg.data, SLMUnitDataConfig) else "?"
        return {"Model type": self.cfg.run.model_type, "Task": f"multi-stream SLM ({num_streams} streams)"}


def train_multistream_slm(cfg: LMConfig) -> None:
    MultiStreamSLMTrainer(cfg).fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    cfg = LMConfig.from_file(args.config)
    if isinstance(cfg.data, SLMUnitDataConfig) and isinstance(cfg.data.source, PreExtractedUnitSource):
        train_multistream_slm(cfg)
    else:
        train_slm(cfg)
