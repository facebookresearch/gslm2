"""CLI: score a trained SLM checkpoint on 2AFC spoken benchmarks (sWuggy/sBLIMP/StoryCloze).

Single-stream models encode benchmark audio on the fly with the frozen training
encoder; multi-stream (GSLM2) models score pre-extracted ``|``-joined unit-pair
manifests; text LMs score tokenized text pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from gslm2.config import LMConfig, OnTheFlyUnitSource, TextDataConfig, load_config
from gslm2.models.factory import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_lm(ckpt_path: Path, model_type: str, model_cfg) -> torch.nn.Module:  # noqa: ANN001
    model = build_model(cfg=model_cfg, model_type=model_type)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    result = model.load_state_dict(state, strict=False)
    assert not result.unexpected_keys, f"unexpected checkpoint keys: {result.unexpected_keys}"
    if result.missing_keys:
        logger.info("missing keys (non-persistent buffers ok): %s", result.missing_keys)
    return model


def _score_single_stream(cfg: LMConfig, args) -> dict[str, dict[str, float]]:  # noqa: ANN001
    from gslm2.eval.slm.native_eval import NativeSlmEvaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_lm(Path(args.checkpoint), cfg.run.model_type, cfg.model).to(device).train(mode=False)
    fe = cfg.data.source.feature_extraction
    encoder = build_model(model_type=fe.model_type, checkpoint=fe.checkpoint).to(device).train(mode=False)
    evaluator = NativeSlmEvaluator(
        model=model,
        encoder=encoder,
        fe=fe,
        device=device,
        batch_size=args.batch_size,
        normalize=cfg.data.source.audio.normalize,
    )
    results = {}
    for bm in args.benchmarks:
        r = evaluator.evaluate_benchmark(Path(args.pairs_dir) / f"{bm}.csv")
        results[bm] = r
        logger.info("%s: acc=%.4f acc_norm=%.4f n=%d", bm, r["accuracy"], r["accuracy_norm"], int(r["npairs"]))
    return results


def _score_multi_stream(cfg: LMConfig, args) -> dict[str, dict[str, float]]:  # noqa: ANN001
    from gslm2.eval.slm.native_eval import NativeMultiStreamSlmEvaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_lm(Path(args.checkpoint), cfg.run.model_type, cfg.model).to(device).train(mode=False)
    evaluator = NativeMultiStreamSlmEvaluator(model=model, device=device, batch_size=args.batch_size)
    results = {}
    for bm in args.benchmarks:
        manifest = Path(args.pairs_dir) / f"{bm}.jsonl"
        if not manifest.is_file():
            manifest = manifest.with_suffix(".csv")
        r = evaluator.evaluate_benchmark(manifest)
        results[bm] = r
        logger.info("%s: acc=%.4f acc_norm=%.4f n=%d", bm, r["accuracy"], r["accuracy_norm"], int(r["npairs"]))
    return results


def _score_text(cfg: LMConfig, args) -> dict[str, dict[str, float]]:  # noqa: ANN001
    import polars as pl

    from gslm2.data.text.dataset import load_tokenizer as _load_tokenizer
    from gslm2.models.core.scoring import sequence_log_likelihood

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_lm(Path(args.checkpoint), cfg.run.model_type, cfg.model).to(device).train(mode=False)
    tokenizer = _load_tokenizer(cfg.data.tokenizer)
    results = {}
    for bm in args.benchmarks:
        pairs = pl.read_csv(Path(args.pairs_dir) / f"{bm}.csv")
        pos = [torch.tensor(tokenizer.encode(t, add_special_tokens=True)) for t in pairs["positive"].to_list()]
        neg = [torch.tensor(tokenizer.encode(t, add_special_tokens=True)) for t in pairs["negative"].to_list()]
        n_correct = n_correct_norm = 0
        for start in range(0, len(pos), args.batch_size):
            ll_c, lln_c = sequence_log_likelihood(model, pos[start : start + args.batch_size], device)
            ll_i, lln_i = sequence_log_likelihood(model, neg[start : start + args.batch_size], device)
            n_correct += int((ll_c > ll_i).sum())
            n_correct_norm += int((lln_c > lln_i).sum())
        r = {"accuracy": n_correct / len(pos), "accuracy_norm": n_correct_norm / len(pos), "npairs": float(len(pos))}
        results[bm] = r
        logger.info("%s: acc=%.4f acc_norm=%.4f n=%d", bm, r["accuracy"], r["accuracy_norm"], len(pos))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Score an SLM checkpoint on 2AFC benchmarks.")
    ap.add_argument("--config", required=True, help="SLM training config YAML (arch + encoder/fe or tokenizer).")
    ap.add_argument("--checkpoint", required=True, help="Trained checkpoint (weights under ['model']).")
    ap.add_argument(
        "--pairs-dir", required=True, help="Benchmark pair manifests (audio CSVs, |-joined units, or text)."
    )
    ap.add_argument("--benchmarks", nargs="+", default=["lexical_test_invocab", "lexical_test_oov", "syntactic_test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    assert isinstance(cfg, LMConfig), f"expected an LMConfig YAML, got {type(cfg).__name__}"
    if isinstance(cfg.data, TextDataConfig):
        results = _score_text(cfg, args)
    elif isinstance(cfg.data.source, OnTheFlyUnitSource):
        results = _score_single_stream(cfg, args)
    else:
        results = _score_multi_stream(cfg, args)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("=== SLM EVAL DONE -> %s ===", args.out)


if __name__ == "__main__":
    main()
