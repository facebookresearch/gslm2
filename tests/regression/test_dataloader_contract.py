"""Dataloader contract: build_dataloader yields batches conforming to the SampleBatch schema."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gslm2.data.speech.dataloader import build_dataloader
from gslm2.data.speech.dataset import SampleBatch

from .helpers import set_deterministic, tiny_data_config, tiny_masking_config, tiny_spidr_config

REQUIRED_KEYS = ("waveforms",)
OPTIONAL_KEYS = ("augmented_waveforms", "phonemes", "languages", "attention_mask", "mask")


def _first_batch(manifest: Path) -> SampleBatch:
    set_deterministic(42)
    data_cfg = tiny_data_config(manifest, enable_padding=True)
    model_cfg = tiny_spidr_config()
    loader = build_dataloader(data_cfg, tiny_masking_config(), model_cfg.extractor_conv_layer_config)
    return next(iter(loader))


def test_dataloader_yields_sample_batch(tiny_manifest: Path) -> None:
    batch = _first_batch(tiny_manifest)
    if not isinstance(batch, SampleBatch):
        raise AssertionError(f"expected SampleBatch, got {type(batch).__name__}")


def test_required_keys_present(tiny_manifest: Path) -> None:
    batch = _first_batch(tiny_manifest)
    for key in REQUIRED_KEYS:
        if getattr(batch, key) is None:
            raise AssertionError(f"required field {key!r} is None")


def test_batch_dimension_consistent(tiny_manifest: Path) -> None:
    batch = _first_batch(tiny_manifest)
    batch_size = batch.waveforms.shape[0]
    if batch_size <= 0:
        raise AssertionError("waveforms has zero batch dim")
    for key in REQUIRED_KEYS + OPTIONAL_KEYS:
        value = getattr(batch, key)
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] != batch_size:
            raise AssertionError(f"batch-dim mismatch for {key!r}: {value.shape[0]} vs {batch_size}")


def test_no_nan_or_inf(tiny_manifest: Path) -> None:
    batch = _first_batch(tiny_manifest)
    for key in REQUIRED_KEYS + OPTIONAL_KEYS:
        value = getattr(batch, key)
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise AssertionError(f"non-finite values in {key!r}")


def test_tensors_non_empty(tiny_manifest: Path) -> None:
    batch = _first_batch(tiny_manifest)
    for key in REQUIRED_KEYS:
        if getattr(batch, key).numel() == 0:
            raise AssertionError(f"empty tensor for {key!r}")


def test_waveform_dtype_and_shape(tiny_manifest: Path) -> None:
    batch = _first_batch(tiny_manifest)
    if batch.waveforms.dim() != 2:
        raise AssertionError(f"waveforms should be 2D [batch, time], got {tuple(batch.waveforms.shape)}")
    if batch.waveforms.dtype != torch.float32:
        raise AssertionError(f"waveforms dtype expected float32, got {batch.waveforms.dtype}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SLM AudioDataset requires GPU runtime")
def test_slm_audio_dataset_schema(tiny_manifest: Path) -> None:
    pytest.importorskip("fairseq2.recipes")
    from gslm2.fs2.unit_lm.recipes.train.dataset_audio import AudioDataset, AudioDatasetConfig

    cfg = AudioDatasetConfig(
        manifest_train=str(tiny_manifest),
        manifest_valid=str(tiny_manifest),
        normalize=True,
        min_audio_length=1000,
        max_audio_length=32000,
        batch_size=2,
        num_prefetch=0,
        num_parallel_calls=1,
    )
    ds = AudioDataset(cfg)
    batch = next(iter(ds.create_train_dataloader(seed=42)))
    if not hasattr(batch, "seqs"):
        raise AssertionError("SLM batch missing .seqs")
    if batch.seqs.dim() != 2:
        raise AssertionError(f"SLM seqs should be 2D, got {tuple(batch.seqs.shape)}")
    if not math.isfinite(batch.seqs.float().abs().sum().item()):
        raise AssertionError("SLM seqs contains non-finite values")
