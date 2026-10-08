"""Online SLM validation runners for the consolidated ``EVAL_REGISTRY``.

Scores an intermediate OPT checkpoint (native ``lm.py`` format) on sLM21
2AFC pair manifests with :class:`NativeSlmEvaluator`, plus dev-NLL
(``lm-loss``) over a LibriLight dev manifest. Both mirror their fairseq2
recipe counterparts (``valid_metrics`` / train-loss scale).

Unit caching reuses the trainer's on-disk conventions
(:func:`save_units` int32 per-file format,
:func:`path_cache_path` stable layout): validation re-scores the same samples
every step, so per-file units are read from ``unit_cache_dir`` when the
encoder fingerprint matches and only misses pay an encoder pass. Fingerprint
covers the frozen encoder config; the training encoder here is always frozen,
so no checkpoint keying is needed (joint training would add one).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gslm2.config import LMConfig, ResumeConfig

import polars as pl
import torch

from gslm2.tools import init_logger

logger = logging.getLogger()

PAIRS_DIR = Path("/datasets/metaspeechssl/shared/assets/manifests/sLM21-dataset/pairs")
DEV_MANIFEST = Path("/datasets/metaspeechssl/shared/assets/manifests/librilight/large_segmented_60s_dev.csv")


def _load_lm_and_encoder(
    lm_cfg: LMConfig, run_model_type: str, checkpoint: str | Path, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build the OPT LM and the frozen unit-extraction encoder for validation."""
    from gslm2.models import build_model

    model = build_model(cfg=lm_cfg.model, model_type=run_model_type)  # type: ignore[arg-type]
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(raw.get("model", raw))
    fe = lm_cfg.data.source.feature_extraction
    encoder = build_model(model_type=fe.model_type, checkpoint=fe.checkpoint)
    return model.to(device).eval(), encoder.to(device).train(mode=False)


def _units_fingerprint(fe: Any, extra: dict[str, Any]) -> str:  # noqa: ANN401
    """Short hash pinning cached units to one encoder configuration."""
    payload = {
        "checkpoint": str(fe.checkpoint),
        "extract_layer": fe.extract_layer,
        "codebook_size": fe.codebook_size,
        "unit_offset": fe.unit_offset,
        "deduplicate": fe.deduplicate,
        "max_token_seq_len": fe.max_token_seq_len,
        **extra,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _read_cached_units(path: Path) -> list[int] | None:
    """One cached unit sequence, or ``None`` on miss or corrupt entry."""
    if not path.exists():
        return None
    try:
        seq = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - stale/partial entry reads as a miss
        return None
    if not isinstance(seq, torch.Tensor) or seq.dim() != 1:
        return None
    return seq.tolist()


def _encode_missing_units(
    evaluator: Any,  # noqa: ANN401 - NativeSlmEvaluator
    missing: list[str],
    cache_dir: Path | None,
    fingerprint: str,
) -> None:
    """Encode uncached paths into ``evaluator.units_cache``, halving the batch on OOM."""
    from gslm2.data.speech.unit_extraction import path_cache_path, save_units

    encode_bs = evaluator.batch_size
    while missing:
        try:
            for path, seq in zip(missing, evaluator.cached_units_for_paths(missing), strict=True):
                evaluator.units_cache[path] = seq.tolist()
                if cache_dir is not None:
                    save_units(path_cache_path(cache_dir, fingerprint, path), seq)
        except torch.cuda.OutOfMemoryError:
            if encode_bs <= 1:
                raise
            encode_bs //= 2
            evaluator.batch_size = encode_bs
            logger.warning("CUDA OOM encoding validation audio, retrying with batch_size=%d", encode_bs)
            torch.cuda.empty_cache()
        else:
            return


def _load_units_cache(
    evaluator: Any,  # noqa: ANN401
    cache_dir: Path | None,
    fingerprint: str,
    paths: list[str],
) -> None:
    """Attach a session cache pre-filled with fingerprint-matching disk entries."""
    from gslm2.data.speech.unit_extraction import path_cache_path

    evaluator.units_cache = {}
    if cache_dir is None:
        return
    for path in dict.fromkeys(paths):
        seq = _read_cached_units(path_cache_path(cache_dir, fingerprint, path))
        if seq is not None:
            evaluator.units_cache[path] = seq


def run_slm_eval(
    spec_config: dict[str, Any],
    resume: ResumeConfig,
    run_dir: Path,
    eval_name: str = "slm",
) -> None:
    """Runner entry: ``(spec.config, resume, cfg.run.dir, spec.resolved_name)``."""
    init_logger()
    from gslm2.config import LMConfig
    from gslm2.eval.runners import _append_metrics
    from gslm2.eval.slm.native_eval import NativeSlmEvaluator

    lm_cfg = LMConfig.from_file(spec_config["lm_config"])
    benchmarks: list[str] = spec_config.get(
        "benchmarks",
        ["lexical_dev_invocab", "lexical_dev_oov", "syntactic_dev", "tSC_dev_expanded"],
    )
    max_pairs: int = spec_config.get("max_pairs", 1000)
    batch_size: int = spec_config.get("batch_size", 32)

    run_model_type: str = spec_config.get("model_type", "opt")
    pairs_dir = Path(spec_config.get("pairs_dir", str(PAIRS_DIR)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, encoder = _load_lm_and_encoder(lm_cfg, run_model_type, resume.checkpoint, device)

    source = lm_cfg.data.source
    fe = source.feature_extraction
    evaluator = NativeSlmEvaluator(
        model=model,
        encoder=encoder,
        fe=fe,
        device=device,
        batch_size=batch_size,
        normalize=source.audio.normalize,
    )

    cache_dir = spec_config.get("unit_cache_dir")
    cache_path = Path(cache_dir) if cache_dir else None
    fingerprint = _units_fingerprint(fe, {"benchmarks": sorted(benchmarks), "max_pairs": max_pairs})

    out_dir = Path(run_dir) / "validation" / f"val-{resume.step}-{eval_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for benchmark in benchmarks:
        pairs = pl.read_csv(pairs_dir / f"{benchmark}.csv")
        if max_pairs:
            pairs = pairs.head(max_pairs)
        manifest = out_dir / f"{benchmark}-max{max_pairs}.csv"
        pairs.write_csv(manifest)
        paths = pairs["path_correct"].to_list() + pairs["path_incorrect"].to_list()
        _load_units_cache(evaluator, cache_path, fingerprint, paths)
        missing = [p for p in dict.fromkeys(paths) if p not in evaluator.units_cache]
        _encode_missing_units(evaluator, missing, cache_path, fingerprint)
        result = evaluator.evaluate_benchmark(manifest)
        _append_metrics(Path(resume.results), resume.step, f"{eval_name}/{benchmark}", result)


def _score_dev_nll(
    model: torch.nn.Module,
    unit_seqs: list[torch.Tensor],
    score_bs: int,
    device: torch.device,
) -> tuple[float, int]:
    """Token-weighted NLL over unit sequences, halving the batch on OOM."""
    from gslm2.models.core.scoring import sequence_log_likelihood

    total_nll, total_tokens = 0.0, 0
    while True:
        try:
            for start in range(0, len(unit_seqs), score_bs):
                chunk = unit_seqs[start : start + score_bs]
                ll, _ = sequence_log_likelihood(model, chunk, device)
                toks = torch.tensor([max(s.numel() - 1, 0) for s in chunk], device=device)
                total_nll += float((-ll).sum())
                total_tokens += int(toks.sum())
        except torch.cuda.OutOfMemoryError:
            if score_bs <= 1:
                raise
            score_bs //= 2
            logger.warning("CUDA OOM scoring dev NLL, retrying with batch_size=%d", score_bs)
            torch.cuda.empty_cache()
        else:
            return total_nll, total_tokens


def run_lm_loss_eval(
    spec_config: dict[str, Any],
    resume: ResumeConfig,
    run_dir: Path,
    eval_name: str = "lm-loss",
) -> None:
    """Dev-NLL runner entry: token-weighted mean next-token NLL (nats/token, same
    scale as ``train/loss``) over the reference valid split."""
    del run_dir
    init_logger()
    from gslm2.config import LMConfig
    from gslm2.eval.runners import _append_metrics
    from gslm2.eval.slm.native_eval import NativeSlmEvaluator

    lm_cfg = LMConfig.from_file(spec_config["lm_config"])
    dev_manifest = Path(spec_config.get("dev_manifest", str(DEV_MANIFEST)))
    score_bs: int = spec_config.get("score_batch_size", spec_config.get("batch_size", 4096))
    # Proven on 1xH100-80GB with full 60 s dev segments: 16/8/4 OOM, 2 fits.
    # The halving retry below remains as a safety net down to 1.
    encode_bs = 2
    cache_dir = spec_config.get("unit_cache_dir")
    cache_path = Path(cache_dir) if cache_dir else None
    max_files: int | None = spec_config.get("max_files")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, encoder = _load_lm_and_encoder(lm_cfg, spec_config.get("model_type", "opt"), resume.checkpoint, device)
    source = lm_cfg.data.source
    fe = source.feature_extraction
    evaluator = NativeSlmEvaluator(
        model=model,
        encoder=encoder,
        fe=fe,
        device=device,
        batch_size=encode_bs,
        normalize=source.audio.normalize,
    )

    paths = pl.read_csv(dev_manifest)["path"].to_list()
    if max_files is not None:
        paths = paths[:max_files]
    fingerprint = _units_fingerprint(fe, {"dev_manifest": str(dev_manifest), "max_files": max_files})
    _load_units_cache(evaluator, cache_path, fingerprint, paths)
    missing = [p for p in dict.fromkeys(paths) if p not in evaluator.units_cache]
    _encode_missing_units(evaluator, missing, cache_path, fingerprint)
    unit_seqs = [torch.tensor(evaluator.units_cache[p], dtype=torch.long) for p in paths]

    total_nll, total_tokens = _score_dev_nll(model, unit_seqs, score_bs, device)
    result = {"dev_loss": total_nll / max(total_tokens, 1)}
    logger.info("dev_loss=%.4f over %d tokens (%s)", result["dev_loss"], float(total_tokens), dev_manifest)
    _append_metrics(Path(resume.results), resume.step, eval_name, result)
