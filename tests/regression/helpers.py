"""Shared helpers for regression tests."""

from __future__ import annotations

import csv
import json
import math
import os
import socket
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from gslm2.config import AUDIO_SAMPLE_RATE, DataConfig, MaskingConfig, SpidRConfig, TextDataConfig
from gslm2.environment import set_seed, setup_pytorch

BASELINES_DIR = Path(__file__).parent / "baselines"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Manifest shape shared by the conftest fixtures and baseline regeneration so a
# committed loss baseline is reproduced by the exact data the test feeds in.
TINY_NUM_SAMPLES = 256
TINY_DURATION_SECONDS = 2.0
TINY_AMPLITUDE = 0.1
TINY_LANGUAGE = "en_arpa"


def write_wav(path: Path, wav: torch.Tensor, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
    """Write a mono [1, time] float waveform as 16-bit PCM WAV.

    Rounds and clips explicitly, pinning the quantization the committed loss baselines
    were generated against.
    """
    quantized = np.clip(np.round(wav.squeeze(0).numpy() * 32768), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(quantized.tobytes())


def read_wav(path: str | Path) -> torch.Tensor:
    """Read a mono WAV as a 1-D float32 tensor of shape [time]."""
    with wave.open(str(path), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0)


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def torchrun_env(workdir: Path, manifest: Path) -> dict[str, str]:
    """Single-rank torchrun env with PYTHONPATH pinned to this checkout's src/."""
    env = os.environ.copy()
    env.update(
        {
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(pick_free_port()),
            "METASR_REGRESSION_MANIFEST": str(manifest),
            "METASR_REGRESSION_WORKDIR": str(workdir),
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")) if p)
    return env


def build_tiny_manifest(audio_dir: Path, manifest_path: Path) -> Path:
    """Write deterministic tiny WAVs + a manifest CSV; returns the manifest path."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(TINY_DURATION_SECONDS * AUDIO_SAMPLE_RATE)
    generator = torch.Generator().manual_seed(0)
    rows = []
    for i in range(TINY_NUM_SAMPLES):
        wav = torch.randn(1, n_frames, generator=generator) * TINY_AMPLITUDE
        path = audio_dir / f"sample_{i:03d}.wav"
        write_wav(path, wav)
        rows.append(
            {"fileid": f"sample_{i:03d}", "path": str(path), "num_frames": n_frames, "language": TINY_LANGUAGE}
        )
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fileid", "path", "num_frames", "language"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def set_deterministic(seed: int = 42) -> None:
    """Seed + deterministic flags without distributed init (CPU-safe)."""
    set_seed(seed)
    # warn_only=True: regression suite tolerates ops that lack deterministic CPU impls.
    setup_pytorch(use_deterministic=False)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def tiny_spidr_config(**overrides: object) -> SpidRConfig:
    base = SpidRConfig(
        extractor_conv_layer_config=[(64, 10, 5), (64, 3, 2), (64, 3, 2)],
        encoder_embed_dim=64,
        encoder_num_layers=2,
        encoder_num_heads=4,
        encoder_ff_interm_features=128,
        encoder_pos_conv_kernel=15,
        encoder_pos_conv_depth=1,
        encoder_pos_conv_groups=4,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_dropout=0.0,
        encoder_dropout=0.0,
        encoder_layer_drop=0.0,
        codebook_size=64,
        num_codebooks=2,
    )
    return replace(base, **overrides) if overrides else base


def tiny_data_config(manifest: Path, **overrides: object) -> DataConfig:
    base = DataConfig(
        manifest=str(manifest),
        num_buckets=4,
        min_sample_size=1_000,
        max_sample_size=32_000,
        max_batch_length=256_000,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
    )
    return replace(base, **overrides) if overrides else base


def tiny_masking_config() -> MaskingConfig:
    return MaskingConfig(mask_prob=0.5, mask_length=4)


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 0.0,
    name: str = "tensor",
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_loss_trajectory_close(
    actual: list[float],
    baseline: list[float],
    *,
    final_tol: float = 0.02,
    step_tol_factor: float = 3.0,
    name: str = "loss",
) -> None:
    if len(actual) != len(baseline):
        raise AssertionError(f"{name}: length mismatch {len(actual)} vs {len(baseline)}")
    # Descent check: the run must actually learn. Auto-calibrated to the baseline's
    # own drop so it needs no magic threshold and catches a frozen/no-op model even
    # when the learning signal is smaller than ``final_tol`` (e.g. SpidR's ~1.7%).
    baseline_descent = baseline[0] - baseline[-1]
    actual_descent = actual[0] - actual[-1]
    if baseline_descent > 0 and actual_descent < 0.5 * baseline_descent:
        raise AssertionError(
            f"{name}: insufficient learning — loss dropped {actual_descent:.4f} "
            f"({actual[0]:.4f}->{actual[-1]:.4f}) vs baseline drop {baseline_descent:.4f}; "
            "model may be frozen or gradients broken"
        )
    final_diff = abs(actual[-1] - baseline[-1]) / max(abs(baseline[-1]), 1e-8)
    if final_diff >= final_tol:
        raise AssertionError(
            f"{name}: final-loss regression: got {actual[-1]:.6f}, "
            f"expected {baseline[-1]:.6f} (rel diff {final_diff:.4f})"
        )
    step_tol = final_tol * step_tol_factor
    for i, (a, b) in enumerate(zip(actual, baseline, strict=True)):
        if not math.isfinite(a):
            raise AssertionError(f"{name}: non-finite value at step {i}: {a}")
        step_diff = abs(a - b) / max(abs(b), 1e-8)
        if step_diff >= step_tol:
            raise AssertionError(f"{name}: diverged at step {i}: {a:.6f} vs {b:.6f} (rel diff {step_diff:.4f})")


def load_json_baseline(path: Path) -> list[float]:
    with path.open() as f:
        return list(json.load(f))


# Gentle LR keeps the 50-step trajectory off zero for stable relative-tolerance parity.
SLM_LM_LR = 1e-4


def slm_small_test_config():
    from gslm2.fs2.models.spidr_opt.config import SpidrOptConfig, SpidrOptEncoderConfig, SpidrOptLMConfig

    return SpidrOptConfig(
        encoder=SpidrOptEncoderConfig(checkpoint="", extract_layer=6, codebook_size=256, dtype=torch.float32),
        lm=SpidrOptLMConfig(
            model_dim=256,
            num_layers=2,
            num_attn_heads=4,
            num_key_value_heads=4,
            ffn_inner_dim=512,
            max_seq_len=513,
        ),
    )


TINY_TEXT_NUM_SAMPLES = 256
TINY_TEXT_VOCAB = [
    "the",
    "cat",
    "sat",
    "on",
    "mat",
    "dog",
    "ran",
    "to",
    "big",
    "red",
    "blue",
    "small",
    "fast",
    "slow",
    "jumped",
    "over",
    "under",
    "beside",
    "happy",
    "sad",
    "a",
    "an",
    "is",
    "was",
    "has",
    "had",
    "will",
    "can",
    "it",
    "he",
    "she",
    "they",
    "we",
    "this",
    "that",
    "and",
    "or",
    "but",
]


def build_tiny_text_manifest(manifest_path: Path) -> Path:
    """Write a deterministic tiny text manifest for LM regression tests."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import random

    rng = random.Random(42)
    lines: list[str] = []
    for _ in range(TINY_TEXT_NUM_SAMPLES):
        length = rng.randint(5, 20)
        sentence = " ".join(rng.choice(TINY_TEXT_VOCAB) for _ in range(length))
        lines.append(sentence)
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def tiny_causal_lm_config(**overrides: object):
    from gslm2.config.models.core import CausalLMConfig

    base = CausalLMConfig(
        model_dim=64,
        max_seq_len=128,
        vocab_size=50272,
        num_layers=2,
        num_attn_heads=4,
        num_key_value_heads=4,
        ffn_inner_dim=128,
        dropout_p=0.0,
    )
    return replace(base, **overrides) if overrides else base


def tiny_text_data_config(manifest: Path, **overrides: object) -> TextDataConfig:
    base = TextDataConfig(
        manifests=[str(manifest)],
        manifests_weights=[1.0],
        tokenizer="facebook/opt-125m",
        task="causal",
        max_length=128,
        batch_size=4,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
    )
    return replace(base, **overrides) if overrides else base


LM_LR = 1e-4


def run_lm_loss_trajectory(manifest: Path, *, steps: int = 50, seed: int = 42) -> list[float]:
    """N-step causal LM training loss on CPU. Returns per-step loss."""
    from torch.optim import AdamW

    from gslm2.data.text.collators import CausalCollator
    from gslm2.data.text.dataset import TextDataset
    from gslm2.models.core.opt import OPT

    set_deterministic(seed)
    cfg = tiny_causal_lm_config()
    model = OPT(cfg)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LM_LR)

    data_cfg = tiny_text_data_config(manifest)
    dataset = TextDataset(data_cfg)
    collate = CausalCollator(pad_token_id=dataset.tokenizer.pad_token_id)

    batch_size = 4
    losses: list[float] = []
    for step in range(steps):
        start = (step * batch_size) % len(dataset)
        samples = [dataset[(start + i) % len(dataset)] for i in range(batch_size)]
        ids, attn, targets = collate(samples)

        loss, _ = model(ids, targets=targets, attention_mask=attn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run_slm_onthefly_loss_trajectory(
    manifest: Path, *, steps: int, seed: int = 42, num_files: int = 8, batch_size: int = 4
) -> list[float]:
    """N-step NEW-STACK on-the-fly SLM training on CPU: random SpidR encoder ->
    ``extract_units`` -> pure-PyTorch OPT causal loss. Mirrors ``SLMTrainer``'s audio
    path. Returns per-step loss. Deterministic given seed + manifest."""
    import csv

    from torch.nn.utils.rnn import pad_sequence
    from torch.optim import AdamW

    from gslm2.config import SpeechUnitConfig, SpidRConfig
    from gslm2.data.common.masking.speech import build_attention_mask
    from gslm2.data.speech.collators import conv_length
    from gslm2.data.speech.unit_extraction import extract_units
    from gslm2.data.text.collators import _IGNORE_INDEX
    from gslm2.models.core.opt import OPT
    from gslm2.models.speech.spidr import SpidR

    set_deterministic(seed)
    encoder = SpidR(SpidRConfig()).train(mode=False)
    encoder.requires_grad_(requires_grad=False)
    fe = SpeechUnitConfig(checkpoint="", extract_layer=6, codebook_size=256, unit_offset=2, max_token_seq_len=512)
    lm = OPT(tiny_causal_lm_config(vocab_size=fe.codebook_size + fe.unit_offset, max_seq_len=fe.max_token_seq_len + 1))
    lm.train()
    optimizer = AdamW(lm.parameters(), lr=SLM_LM_LR)

    with manifest.open() as f:
        rows = list(csv.DictReader(f))[:num_files]
    waveforms = [read_wav(row["path"]) for row in rows]
    conv = encoder.cfg.extractor_conv_layer_config

    losses: list[float] = []
    for step in range(steps):
        start = (step * batch_size) % len(waveforms)
        batch = [waveforms[(start + i) % len(waveforms)] for i in range(batch_size)]
        wf = pad_sequence(batch, batch_first=True)
        lens = torch.tensor([w.size(0) for w in batch])
        attention_mask = build_attention_mask(conv_length(conv, lens))
        tokens, _ = extract_units(encoder, wf, attention_mask, fe)
        pad = tokens == 0
        targets = torch.full_like(tokens, _IGNORE_INDEX)
        targets[:, :-1] = tokens[:, 1:].masked_fill(pad[:, 1:], _IGNORE_INDEX)
        loss, _ = lm(tokens, targets=targets, attention_mask=(~pad).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run_lm_loss_trajectory_packed(manifest: Path, *, steps: int = 50, seed: int = 42) -> list[float]:
    """N-step packed causal LM training loss on CPU. Returns per-step loss."""
    from torch.optim import AdamW

    from gslm2.data.text.collators import PackedCausalCollator
    from gslm2.data.text.dataset import TextDataset
    from gslm2.models.core.opt import OPT

    set_deterministic(seed)
    cfg = tiny_causal_lm_config()
    model = OPT(cfg)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LM_LR)

    data_cfg = tiny_text_data_config(manifest)
    dataset = TextDataset(data_cfg)
    collate = PackedCausalCollator(pad_token_id=dataset.tokenizer.pad_token_id, max_length=data_cfg.max_length)

    batch_size = 4
    losses: list[float] = []
    for step in range(steps):
        start = (step * batch_size) % len(dataset)
        samples = [dataset[(start + i) % len(dataset)] for i in range(batch_size)]
        packed_ids, doc_ids, position_ids, targets = collate(samples)

        loss, _ = model(packed_ids, targets=targets, position_ids=position_ids, doc_ids=doc_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run_slm_loss_trajectory(
    manifest: Path, *, steps: int, seed: int = 42, num_files: int = 8, batch_size: int = 4
) -> list[float]:
    """N-step SLM (SpidrOpt) training loss on CPU, no fairseq2 recipe harness.

    Builds the small_test model, encodes fixed audio through the frozen SpidR
    encoder, and trains the OPT LM with its native auto-regressive loss. Returns
    the per-step training loss. Deterministic given seed + manifest.
    """
    from fairseq2.datasets import SequenceBatch
    from fairseq2.nn import BatchLayout
    from torch.optim import AdamW

    from gslm2.fs2.models.spidr_opt.factory import create_spidr_opt_model

    set_deterministic(seed)
    model = create_spidr_opt_model(slm_small_test_config(), device=torch.device("cpu"))
    model.train()
    optimizer = AdamW([p for p in model.lm.parameters() if p.requires_grad], lr=SLM_LM_LR)

    with manifest.open() as f:
        rows = list(csv.DictReader(f))[:num_files]
    waveforms = [read_wav(row["path"]) for row in rows]
    batches = [
        torch.nn.utils.rnn.pad_sequence(waveforms[i : i + batch_size], batch_first=True)
        for i in range(0, len(waveforms), batch_size)
    ]

    losses: list[float] = []
    for step in range(steps):
        audio = batches[step % len(batches)]
        tokens, seq_lens, _ = model.encode(audio, BatchLayout.of(audio))
        input_batch, target_batch = SequenceBatch(tokens, seq_lens.tolist()).as_auto_regressive()
        seqs, seqs_layout = input_batch.as_input()
        seq_lens_list = (
            seqs_layout.seq_lens if isinstance(seqs_layout.seq_lens, list) else seqs_layout.seq_lens.tolist()
        )
        max_len = seqs.size(1)
        attn_mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(seq_lens_list).unsqueeze(1)
        loss, _ = model.lm(seqs, targets=target_batch.seqs, attention_mask=attn_mask.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
