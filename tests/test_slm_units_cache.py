"""Guards on the native SLM validation unit caches.

Validation re-scores the same samples every step (fixed pair manifests via
``head(max_pairs)``, frozen dev manifest), so per-file units must be cached and
reused — otherwise every 1k-step validation re-pays a full encoder pass, which
OOMs on 60 s dev segments above batch 2 on 1xH100.

Entries are per-file ``.pt`` tensors in the trainer's int32 format
(:func:`save_units`), keyed by audio-path hash under a fingerprint namespace
(:func:`path_cache_path`), so they stay valid across dev/pair manifests and
coexist with the trainer's index-keyed entries in one cache dir.
Run from the worktree root: ``PYTHONPATH=src python -m pytest tests/test_slm_units_cache.py``
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from gslm2.data.speech.unit_extraction import path_cache_path
from gslm2.eval.slm.native_eval import NativeSlmEvaluator
from gslm2.eval.slm.validation_runners import _load_units_cache, _read_cached_units, _units_fingerprint


def _fe(**overrides: object) -> SimpleNamespace:
    base = {
        "checkpoint": "/ckpt/step_0.pt",
        "extract_layer": 7,
        "codebook_size": 256,
        "unit_offset": 2,
        "deduplicate": True,
        "max_token_seq_len": 2048,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_fingerprint_stable_for_same_config() -> None:
    extra = {"benchmarks": ["lexical_dev_invocab"], "max_pairs": 5000}
    assert _units_fingerprint(_fe(), extra) == _units_fingerprint(_fe(), extra)


def test_fingerprint_changes_with_encoder_and_scope() -> None:
    extra = {"benchmarks": ["lexical_dev_invocab"], "max_pairs": 5000}
    base = _units_fingerprint(_fe(), extra)
    assert base != _units_fingerprint(_fe(extract_layer=8), extra)
    assert base != _units_fingerprint(_fe(checkpoint="/ckpt/step_5000.pt"), extra)
    assert base != _units_fingerprint(_fe(), {"benchmarks": ["lexical_dev_invocab"], "max_pairs": 1000})
    assert base != _units_fingerprint(_fe(), {"benchmarks": ["syntactic_dev"], "max_pairs": 5000})


def test_path_cache_layout(tmp_path: Path) -> None:
    fp = "abc123"
    a = path_cache_path(tmp_path, fp, "/audio/a.wav")
    assert a == path_cache_path(tmp_path, fp, "/audio/a.wav")
    assert a != path_cache_path(tmp_path, fp, "/audio/b.wav")
    assert a != path_cache_path(tmp_path, "otherfp", "/audio/a.wav")
    # Hex digests never collide with the trainer's numeric index entries.
    assert a.name != "12.pt"
    assert a.parent.name == fp


def test_disk_roundtrip_and_corrupt_entry(tmp_path: Path) -> None:
    from gslm2.data.speech.unit_extraction import save_units

    p = path_cache_path(tmp_path, "fp", "/audio/a.wav")
    assert _read_cached_units(p) is None  # miss
    save_units(p, torch.tensor([4, 5, 6], dtype=torch.long))
    assert _read_cached_units(p) == [4, 5, 6]
    p.write_bytes(b"not a tensor")
    assert _read_cached_units(p) is None  # corrupt reads as a miss


def _evaluator(**overrides: object) -> NativeSlmEvaluator:
    base = {
        "model": None,
        "encoder": None,
        "fe": None,
        "device": torch.device("cpu"),
        "batch_size": 2,
        "units_cache": {},
    }
    base.update(overrides)
    return NativeSlmEvaluator(**base)  # type: ignore[arg-type]


def test_cached_units_for_paths_serves_hits_without_encoding() -> None:
    """A warm cache must never touch the encoder."""
    evaluator = _evaluator(units_cache={"a": [1, 2, 3]})

    def _fail(_paths: list[str]) -> list[torch.Tensor]:
        raise AssertionError("encoder called on cache hit")

    evaluator._units_for_paths = _fail  # type: ignore[method-assign]
    (seq,) = evaluator.cached_units_for_paths(["a"])
    assert seq.tolist() == [1, 2, 3]


def test_cached_units_for_paths_encodes_missing_once_then_reuses() -> None:
    """Misses are encoded once and served from cache afterwards."""
    calls: list[list[str]] = []
    evaluator = _evaluator(units_cache={})

    def _fake(paths: list[str]) -> list[torch.Tensor]:
        calls.append(paths)
        return [torch.tensor([len(p), 7], dtype=torch.long) for p in paths]

    evaluator._units_for_paths = _fake  # type: ignore[method-assign]
    first = evaluator.cached_units_for_paths(["a", "b", "a"])
    second = evaluator.cached_units_for_paths(["a", "b"])
    assert [s.tolist() for s in first] == [[1, 7], [1, 7], [1, 7]]
    assert [s.tolist() for s in second] == [[1, 7], [1, 7]]
    assert calls == [["a", "b"]]


def test_cached_units_for_paths_disabled_cache_encodes_every_time() -> None:
    """``units_cache=None`` keeps the old always-encode behavior."""
    calls = 0
    evaluator = _evaluator(units_cache=None)

    def _fake(paths: list[str]) -> list[torch.Tensor]:
        nonlocal calls
        calls += 1
        return [torch.tensor([5], dtype=torch.long) for _ in paths]

    evaluator._units_for_paths = _fake  # type: ignore[method-assign]
    evaluator.cached_units_for_paths(["a"])
    evaluator.cached_units_for_paths(["a"])
    assert calls == 2


def test_load_units_cache_prefills_from_disk(tmp_path: Path) -> None:
    """Disk entries matching the fingerprint warm the session cache."""
    from gslm2.data.speech.unit_extraction import save_units

    fp = _units_fingerprint(_fe(), {"benchmarks": ["b"], "max_pairs": 10})
    save_units(path_cache_path(tmp_path, fp, "/audio/a.wav"), torch.tensor([9, 9], dtype=torch.long))
    evaluator = _evaluator()
    _load_units_cache(evaluator, tmp_path, fp, ["/audio/a.wav", "/audio/missing.wav"])
    assert evaluator.units_cache == {"/audio/a.wav": [9, 9]}
