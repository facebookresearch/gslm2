"""Functional coverage for the pure-PyTorch multi-stream OPT + its sLM21 scoring.

fairseq2 forward/loss parity lives in tests/regression/test_slm_multistream_forward_parity.py (it
needs the fairseq2 reference and skips in CI on main_mm_clean); these checks are pure PyTorch and run
everywhere: shapes, a finite decreasing loss with every parameter receiving a gradient, the single-
stream and custom-generation-order edge cases, and the multi-stream 2AFC scoring path.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import fields

import pytest
import torch

from gslm2.config import (
    CausalLMConfig,
    LMConfig,
    MultiStreamCausalLMConfig,
    PreExtractedUnitSource,
    SLMUnitDataConfig,
)
from gslm2.data.speech.multistream import (
    MultiStreamUnitCollator,
    MultiStreamUnitDataset,
    MultiStreamUnitPackedCollator,
)
from gslm2.eval.slm.native_eval import NativeMultiStreamSlmEvaluator, _parse_multistream_units
from gslm2.models import build_model
from gslm2.models.core.multi_stream_opt import MultiStreamOPT
from gslm2.models.core.scoring import multistream_sequence_log_likelihood

_VOCAB = (16, 12, 8)


def _tiny(**overrides: object) -> MultiStreamCausalLMConfig:
    base = {
        "vocab_sizes": _VOCAB,
        "model_dim": 32,
        "max_seq_len": 64,
        "num_layers": 2,
        "num_attn_heads": 4,
        "num_key_value_heads": 4,
        "ffn_inner_dim": 64,
        "dropout_p": 0.0,
        "depformer_model_dim": 16,
        "depformer_num_layers": 2,
        "depformer_num_attn_heads": 4,
        "depformer_num_key_value_heads": 4,
        "depformer_ffn_inner_dim": 32,
    }
    return MultiStreamCausalLMConfig(**{**base, **overrides})


def _streams(vocab: tuple[int, ...], batch: int = 2, seq_len: int = 10) -> torch.Tensor:
    return torch.stack([torch.randint(2, v, (batch, seq_len)) for v in vocab], dim=1)  # [B, S, T]


def test_multistream_inference_shapes() -> None:
    torch.manual_seed(0)
    model = MultiStreamOPT(_tiny())
    logits = model(_streams(_VOCAB))
    assert [tuple(x.shape) for x in logits] == [(2, 10, v) for v in _VOCAB]


def test_multistream_training_loss_and_grads() -> None:
    torch.manual_seed(0)
    model = MultiStreamOPT(_tiny())
    streams = _streams(_VOCAB)
    targets = torch.full_like(streams, 1)
    targets[:, :, :-1] = streams[:, :, 1:]
    loss, log_dict = model(streams, targets=targets, stream_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # Training reports the two terms the loss blends alongside the total.
    assert set(log_dict) == {"stream_specific_loss", "gen_loss"}
    assert all(v.ndim == 0 and torch.isfinite(v) for v in log_dict.values())
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad (DDP would flag these): {missing}"


@pytest.mark.parametrize("init_scheme", ["fairseq2", "standard"])
def test_multistream_optimizes(init_scheme: str) -> None:
    torch.manual_seed(0)
    model = MultiStreamOPT(_tiny(init_scheme=init_scheme))
    streams = _streams(_VOCAB)
    targets = torch.full_like(streams, 1)
    targets[:, :, :-1] = streams[:, :, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss, _ = model(streams, targets=targets, stream_idx=0)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_multistream_standard_init_scheme() -> None:
    """``init_scheme='standard'`` scales the output heads by input width and rescales residual blocks.

    Both head stacks take N(0, in_dim^-0.5) rather than trunc_normal(init_std), and each residual stack's
    output projections are divided by sqrt(2 * (layer_id + 1)) to its own depth, while the query, key,
    value and feed-forward input projections keep ``init_std``. Expected values are the analytic forms of
    those initialisers; ``init_std`` is set well away from either head scale to keep them apart.
    """
    torch.manual_seed(0)
    cfg = _tiny(init_std=0.2, init_scheme="standard")
    model = MultiStreamOPT(cfg)
    for stream, proj in enumerate(model.final_proj):
        assert proj.weight.std().item() == pytest.approx(cfg.model_dim**-0.5, rel=0.2), f"final_proj[{stream}]"
    for stream, proj in enumerate(model.depformer_final_proj):
        expected = cfg.depformer_model_dim**-0.5
        assert proj.weight.std().item() == pytest.approx(expected, rel=0.2), f"depformer_final_proj[{stream}]"
    assert model.depformer_bos.std().item() == pytest.approx(cfg.depformer_model_dim**-0.5, rel=0.3)
    # Everything else keeps the generic scale, and the padding row stays zeroed.
    assert model.stream_embed[0].weight.std().item() == pytest.approx(cfg.init_std, rel=0.2)
    assert model.backbone.layers[0].attn.q_proj.weight.std().item() == pytest.approx(cfg.init_std, rel=0.2)
    assert model.stream_embed[0].weight[cfg.pad_idx].abs().sum().item() == 0.0
    # Residual rescaling applies per stack depth, and only to the projections writing back into the
    # residual stream; the input-side projections are untouched.
    for stack in (model.backbone, model.depformer):
        for layer_id, layer in enumerate(stack.layers):
            scale = cfg.init_std / math.sqrt(2.0 * (layer_id + 1))
            assert layer.attn.out_proj.weight.std().item() == pytest.approx(scale, rel=0.2), f"out_proj {layer_id}"
            assert layer.ffn.fc2.weight.std().item() == pytest.approx(scale, rel=0.2), f"fc2 {layer_id}"
            assert layer.ffn.fc1.weight.std().item() == pytest.approx(cfg.init_std, rel=0.2), f"fc1 {layer_id}"


def test_multistream_fairseq2_init_scheme() -> None:
    """``init_scheme='fairseq2'`` sets N(0, 1) embeddings and xavier-uniform attention projections.

    Token, depformer and position embeddings are N(0, 1); attention projections are xavier-uniform with
    gain ``2**-0.5`` on query, key and value and ``1.0`` on the output; feed-forward projections keep
    PyTorch's default kaiming-uniform bound of 1/sqrt(3 * fan_in). Expected values are the analytic forms
    of those initialisers, so this needs no reference implementation.
    """
    torch.manual_seed(0)
    cfg = _tiny(init_scheme="fairseq2")
    model = MultiStreamOPT(cfg)

    for embed in (*model.stream_embed, *model.depformer_embed):
        assert embed.weight.std().item() == pytest.approx(1.0, rel=0.2)
    assert model.backbone.pos_embed.weight.weight.std().item() == pytest.approx(1.0, rel=0.2)
    for stack, width in ((model.backbone, cfg.model_dim), (model.depformer, cfg.depformer_model_dim)):
        for layer in stack.layers:
            attn = layer.attn
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                assert proj.weight.std().item() == pytest.approx((2 * width) ** -0.5, rel=0.2)
            assert attn.out_proj.weight.std().item() == pytest.approx(width**-0.5, rel=0.2)
            # PyTorch's default kaiming-uniform bound gives std = 1/sqrt(3 * fan_in).
            for proj in (layer.ffn.fc1, layer.ffn.fc2):
                assert proj.weight.std().item() == pytest.approx((3 * proj.in_features) ** -0.5, rel=0.2)
    assert model.final_proj[0].weight.std().item() == pytest.approx(cfg.model_dim**-0.5, rel=0.2)
    expected = cfg.depformer_model_dim**-0.5
    assert model.depformer_final_proj[0].weight.std().item() == pytest.approx(expected, rel=0.2)
    assert model.stream_embed[0].weight[cfg.pad_idx].abs().sum().item() == 0.0


def test_multistream_config_matches_causal_lm_decoder_fields() -> None:
    """Decoder fields shared with ``CausalLMConfig`` carry the same name, type and default.

    ``MultiStreamCausalLMConfig`` declares the decoder fields it uses instead of inheriting them, which
    keeps fields that only a single-stream model can use off its surface, at the cost of writing the
    shared ones out twice. This pins the two declarations together, so a field added to one is not
    silently missing from the other, and names the fields that belong to only one of them.
    """
    single_stream_only = {"vocab_size", "tie_word_embeddings"}
    # Each model defaults to the initialisation its own architecture was tuned against.
    divergent_defaults = {"init_scheme"}

    shared = {f.name: f for f in fields(CausalLMConfig) if f.name not in single_stream_only}
    multi_stream = {f.name: f for f in fields(MultiStreamCausalLMConfig)}

    assert not shared.keys() - multi_stream.keys(), (
        f"decoder fields missing from MultiStreamCausalLMConfig: {sorted(shared.keys() - multi_stream.keys())}"
    )
    for name, field in shared.items():
        assert multi_stream[name].type == field.type, f"{name}: type differs between the two configs"
        if name not in divergent_defaults:
            assert multi_stream[name].default == field.default, f"{name}: default differs between the two configs"
    leaked = sorted(single_stream_only & multi_stream.keys())
    assert not leaked, f"fields only a single-stream model can use reached MultiStreamCausalLMConfig: {leaked}"


def test_multistream_edge_cases() -> None:
    torch.manual_seed(0)
    single = MultiStreamOPT(_tiny(vocab_sizes=(16,), depformer_num_layers=1))
    s = _streams((16,))
    t = torch.full_like(s, 1)
    t[:, :, :-1] = s[:, :, 1:]
    assert len(single(s)) == 1
    assert single(s, targets=t, stream_idx=0)[0].ndim == 0

    reordered = MultiStreamOPT(_tiny(depformer_generation_order=(2, 0, 1)))
    assert len(reordered(_streams(_VOCAB))) == 3


def test_multistream_scoring_and_2afc(tmp_path) -> None:
    torch.manual_seed(0)
    model = MultiStreamOPT(_tiny())
    model.train(mode=False)

    assert _parse_multistream_units("2 3 4|5 6 7|8 9 10").shape == (3, 3)

    seqs = [torch.stack([torch.randint(2, v, (length,)) for v in _VOCAB]) for length in (6, 9, 5)]
    raw, norm = multistream_sequence_log_likelihood(model, seqs, torch.device("cpu"))
    assert raw.shape == (3,)
    assert norm.shape == (3,)
    assert torch.isfinite(raw).all()

    def _row() -> dict[str, str]:
        def _one() -> str:
            return "|".join(" ".join(str(int(x)) for x in torch.randint(2, v, (7,))) for v in _VOCAB)

        return {"positive": _one(), "negative": _one()}

    manifest = tmp_path / "tSC_dev.jsonl"
    manifest.write_text("\n".join(json.dumps(_row()) for _ in range(8)) + "\n")
    scorer = NativeMultiStreamSlmEvaluator(model=model, device=torch.device("cpu"), batch_size=4)
    result = scorer.evaluate_benchmark(manifest)
    assert result["npairs"] == 8.0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["accuracy_norm"] <= 1.0


def _write_unit_manifest(path, seq_lens: tuple[int, ...], offset: int = 2) -> None:
    rng = random.Random(0)
    lines = []
    for length in seq_lens:  # each line = per-stream blocks concatenated; raw ids fit each stream's vocab
        parts: list[int] = []
        for vocab in _VOCAB:
            parts += [rng.randint(0, vocab - offset - 1) for _ in range(length)]
        lines.append(" ".join(str(x) for x in parts))
    path.write_text("\n".join(lines) + "\n")


def test_multistream_unit_dataset_and_collator(tmp_path) -> None:
    manifest = tmp_path / "units_0.txt"
    _write_unit_manifest(manifest, seq_lens=(6, 8, 6, 8))
    source = PreExtractedUnitSource(
        manifests=[str(manifest)],
        manifests_weights=[1.0],
        unit_offset=2,
        pad_idx=1,
        min_seq_len=2,
        max_seq_len=64,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
        drop_last=False,
    )
    dataset = MultiStreamUnitDataset(source, num_streams=len(_VOCAB))
    assert len(dataset) == 4
    streams, targets, mask = MultiStreamUnitCollator(num_streams=len(_VOCAB), pad_idx=1)([dataset[0], dataset[1]])
    assert streams.shape == (2, len(_VOCAB), 8)
    assert targets.shape == streams.shape
    assert mask[0].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    for stream, vocab in enumerate(_VOCAB):
        assert int(streams[:, stream].max()) < vocab  # per-stream tokens fit that stream's vocab

    model = MultiStreamOPT(_tiny())
    loss, _ = model(streams, targets=targets, attention_mask=mask, stream_idx=0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multistream_config_from_file(tmp_path) -> None:
    config = tmp_path / "ms.yaml"
    config.write_text(
        "run:\n"
        "  workdir: /tmp/ms\n"
        "  wandb_project: test\n"
        "  model_type: msopt\n"
        "model:\n"
        "  vocab_sizes: [16, 12, 8]\n"
        "  model_dim: 32\n"
        "  num_layers: 2\n"
        "  num_attn_heads: 4\n"
        "  num_key_value_heads: 4\n"
        "  ffn_inner_dim: 64\n"
        "  depformer_generation_order: [2, 0, 1]\n"
        "data:\n"
        "  num_streams: 3\n"
        "  source:\n"
        "    unit_offset: 2\n"
        "    manifests: [/tmp/units_0.txt]\n"
        "    manifests_weights: [1.0]\n"
        "    batch_size: 8\n"
        "optimizer: {}\n"
    )
    cfg = LMConfig.from_file(config)
    assert isinstance(cfg.data, SLMUnitDataConfig)
    assert isinstance(cfg.model, MultiStreamCausalLMConfig)
    assert cfg.data.num_streams == 3
    assert cfg.model.num_streams == 3
    assert type(build_model(cfg=cfg.model, model_type=cfg.run.model_type)).__name__ == "MultiStreamOPT"


def test_multistream_packing_matches_padded() -> None:
    """Sequence packing is a throughput optimization, not a different objective: the packed loss
    (several documents per row, block-diagonal attention) must equal the padded loss per document."""
    torch.manual_seed(0)
    model = MultiStreamOPT(_tiny())
    model.train(mode=False)  # deterministic; dropout is already 0 in _tiny
    streams_n, pad, la, lb = len(_VOCAB), model.pad_idx, 6, 8
    doc_a = torch.stack([torch.randint(2, v, (la,)) for v in _VOCAB])  # [S, la]
    doc_b = torch.stack([torch.randint(2, v, (lb,)) for v in _VOCAB])  # [S, lb]

    # padded: one row per document
    t = max(la, lb)
    padded = torch.full((2, streams_n, t), pad)
    padded[0, :, :la] = doc_a
    padded[1, :, :lb] = doc_b
    pmask = torch.zeros(2, t, dtype=torch.long)
    pmask[0, :la] = 1
    pmask[1, :lb] = 1
    ptargets = torch.full_like(padded, pad)
    ptargets[:, :, :-1] = padded[:, :, 1:]

    # packed: both documents in one row, built by the collator
    collated = MultiStreamUnitPackedCollator(num_streams=streams_n, max_len=64, pad_idx=pad)(
        [doc_a.reshape(-1), doc_b.reshape(-1)]
    )
    streams, targets, kmask, doc_ids = collated
    assert streams.shape[0] == 1  # both documents packed into a single row
    assert doc_ids[0, :la].tolist() == [0] * la
    assert doc_ids[0, la : la + lb].tolist() == [1] * lb

    for k in range(streams_n):
        padded_loss, _ = model(padded, targets=ptargets, attention_mask=pmask, stream_idx=k)
        packed_loss, _ = model(streams, targets=targets, attention_mask=kmask, doc_ids=doc_ids, stream_idx=k)
        torch.testing.assert_close(packed_loss, padded_loss, atol=1e-4, rtol=1e-3)
