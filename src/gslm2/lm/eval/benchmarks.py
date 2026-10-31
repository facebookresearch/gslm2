# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

# Always call init_fairseq2() before using any fairseq2 functionality
from fairseq2 import init_fairseq2
from fairseq2.device import CPU, Device
from fairseq2.gang import FakeGang, ProcessGroupGang
from fairseq2.nn.utils.module import to_device
from fairseq2.utils.rng import RngBag

from gslm2.lm.config import EvalConfig, get_slm_config
from gslm2.lm.eval.metrics import SLMEvalMetrics, cleanup_distributed, eval_lm, setup_distributed
from gslm2.lm.inference import build_opt_config, parse_model_overrides
from gslm2.lm.inference import get_single_ckpt_path as get_single_ckpt_path
from gslm2.lm.models.opt import create_opt_model
from gslm2.shared.distributed import DistributedContext
from gslm2.shared.utils.typing import PathLike

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout
)
logger = logging.getLogger(__name__)


def slm_eval_model(  # noqa: PLR0913
    ckpt_path: PathLike,
    metric_path: PathLike,
    metric_names: list[str],
    output_results_file: PathLike,
    max_pairs: int | None = None,
    batch_size: int | None = 32,
    seed: int = 0,
    vocab_size: int | list[int] = 256,
    offset: int = 2,
    bos: int | None = None,
    dist_context: DistributedContext | None = None,
    local_rank: int = -1,
    metadata: dict | None = None,
    model_preset: str = "opt125m",
    distill_temperature: float = 1.0,
    depformer_generation_order: list[int] | None = None,
    model_config_overrides: dict | None = None,
) -> tuple[dict[str, object], DistributedContext]:
    """
    Evaluate a fairseq2 model on a set of metrics

    Args:
        ckpt_path (str or Path): Path to the model checkpoint
        metric_path (str or Path): Path to the metric directory
        metric_names (list[str]): List of metric names
        max_pairs (int, optional): Maximum number of pairs to evaluate
        batch_size (int, optional): Batch size for evaluation
        seed (int, optional): Random seed. Only impactful if max_pairs is set
        vocab_size (int, optional): Vocabulary size of the model
        offset (int, optional): Token offset
        bos (int, optional): Beginning of sequence token
        local_rank (int, optional): Local rank for distributed training

    Returns:
        dict: Results of the evaluation
    """
    # Setup distributed training with local_rank
    if metadata is None:
        metadata = {}
    if dist_context is None:
        dist_context = setup_distributed(local_rank)
    rank, world_size, local_rank = dist_context.rank, dist_context.world_size, dist_context.local_rank
    is_slurm = dist_context.mode == "slurm"
    is_main_process = rank == 0

    # Set device
    device = Device(f"cuda:{local_rank}") if torch.cuda.is_available() else CPU

    # setup single fairseq2 DP gang
    # Get the default process group that was initialized in setup_distributed
    process_group = dist.group.WORLD
    if world_size > 1 and process_group is None:
        raise RuntimeError("distributed benchmark evaluation requires an initialized process group")
    gang = (
        ProcessGroupGang(process_group, device) if world_size > 1 and process_group is not None else FakeGang(device)
    )

    # Set random seed for reproducibility
    default_rng_bag = RngBag.from_device_defaults(CPU, device)
    default_rng_bag.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    # Convert paths to Path objects
    ckpt_path = Path(ckpt_path)
    metric_path = Path(metric_path)

    output_results_file = Path(output_results_file)
    output_results_file.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"slm_ckpt": str(ckpt_path)}
    results.update(metadata)

    # Create model with the right architecture and vocab size
    overrides = (
        {"distill_temperature": distill_temperature, "depformer_generation_order": depformer_generation_order}
        if isinstance(vocab_size, list)
        else {}
    )
    overrides.update(model_config_overrides or {})
    opt_config = build_opt_config(model_preset, vocab_size, offset, model_config_overrides=overrides)
    model = create_opt_model(opt_config)

    # Load checkpoint
    if is_main_process:
        logger.info("Loading checkpoint from %s", ckpt_path)
        logger.info(opt_config)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # This handles both the old fs2 checkpoint format, and the v0.5+ one
    state_dict = ckpt.get("model") or ckpt
    model.load_state_dict(state_dict)
    model.eval()
    to_device(model, device)

    metric_paths = [metric_path / f"{metric_name}.jsonl" for metric_name in metric_names]

    metrics = SLMEvalMetrics(metric_paths, gang)

    metric_results = eval_lm(
        model,
        metrics,
        gang,
        max_pairs=max_pairs,
        batch_size=batch_size,
        dataloader_num_workers=1 if is_slurm else 2,
        seed=seed,
        offset=offset,
        bos=bos,
    )

    metric_floats = {k: v.item() for k, v in metric_results.items()}  # type: ignore  # noqa: PGH003
    results.update(metric_floats)

    # Log overall results from main process only
    if is_main_process:
        logger.info("Overall results:")
        logger.info("%s", str(metric_floats))

        # Dump results to jsonl file
        logger.info("Saving scores to %s", output_results_file)
        with Path(output_results_file).open("a") as file:
            json.dump(results, file)
            file.write("\n")

    return metric_results, dist_context


def get_best_ckpt_paths(
    common_folder: Path,
    feature_dir_override: Path | None = None,
    dist_context: DistributedContext | None = None,
) -> list[Path]:
    # Fall back on feature override folder if provided,
    # This is useful when the eval is the only step invoked,
    # So common_folder doesn't even contain a relevant model checkpoint.
    ckpt_dir = common_folder / "ckpt"
    if not ckpt_dir.exists():
        if feature_dir_override is not None:
            ckpt_dir = feature_dir_override / "ckpt"
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Neither {common_folder / 'ckpt'} nor {ckpt_dir} exist.")
        else:
            raise FileNotFoundError(f"{ckpt_dir} does not exist and no feature_override_folder provided.")

    # This is created by fairseq2, e.g. "ws_8.399a7ee8"
    if (ckpt_dir / "checkpoints").exists():
        subfolder = ckpt_dir
    else:
        try:
            subfolder = next(f for f in ckpt_dir.iterdir() if f.is_dir())
        except StopIteration:
            raise FileNotFoundError(f"No checkpoint subdirectories found in {ckpt_dir}") from None
    result_paths = []

    # Only real step dirs: skips scores/, unified_ckpt/ and partial step_N.tmp writes.
    for f in (subfolder / "checkpoints").iterdir():
        if f.is_dir() and re.fullmatch(r"step_\d+", f.name):
            model_pt_file = get_single_ckpt_path(f, dist_context=dist_context)
            result_paths.append(model_pt_file.absolute())

    return result_paths


def slm_eval_models(cfg: EvalConfig) -> None:
    """Wrapper function to evaluate multiple SLM checkpoints
    For instance if one want to evaluate the best and the last checkpoints
    """

    if cfg.eval_pairs_dir is not None:
        metric_base_path = cfg.eval_pairs_dir
    else:
        metric_base_path = cfg.feature_dir_override if cfg.feature_dir_override else cfg.common_folder
    metric_path = metric_base_path / "test" / "pairs"

    dist_context = setup_distributed(cfg.local_rank)

    ckpt_paths = get_best_ckpt_paths(
        cfg.common_folder,
        feature_dir_override=cfg.feature_dir_override,
        dist_context=dist_context,
    )

    logger.info("Evaluating %s SLM models", str(len(ckpt_paths)))

    for ckpt_path in ckpt_paths:
        _, dist_context = slm_eval_model(
            ckpt_path=ckpt_path,
            metric_path=metric_path,
            metric_names=cfg.metric_names,
            output_results_file=cfg.output_results_file,
            max_pairs=cfg.max_pairs,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            vocab_size=cfg.vocab_size,
            offset=cfg.offset,
            bos=cfg.bos,
            dist_context=dist_context,
            local_rank=cfg.local_rank,
            metadata=cfg.metadata,
            model_preset=cfg.model_preset,
            distill_temperature=cfg.distill_temperature,
            depformer_generation_order=cfg.depformer_generation_order,
            model_config_overrides=cfg.model_config_overrides,
        )

    if cfg.delete_data_model_after_eval and (dist_context is None or dist_context.rank == 0):
        assert metric_base_path.parent.name.startswith("SLM_")
        logger.info("Evaluation done. Deleting the folder %s", metric_base_path.parent)
        shutil.rmtree(metric_base_path.parent)

    # Cleanup distributed context
    cleanup_distributed()


def main() -> None:
    """Score fairseq2 speech-unit or text LM checkpoints on tokenized pairs."""
    parser = argparse.ArgumentParser(description="Evaluate a fairseq2 language model on tokenized benchmark pairs.")
    parser.add_argument("--config-file", type=Path, help="Standalone fairseq2 recipe YAML.")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint file or fairseq2 step directory.")
    parser.add_argument("--pairs-dir", type=Path, help="Directory containing benchmark JSONL pair manifests.")
    parser.add_argument(
        "--benchmarks", nargs="+", default=["lexical_test_invocab", "lexical_test_oov", "syntactic_test"]
    )
    parser.add_argument("--out", type=Path, help="JSONL results file.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--config_path", type=Path, help="Full extraction/train/eval pipeline YAML.")
    parser.add_argument("--feature_dir", type=Path, help="Pre-extracted features for a pipeline config.")
    args = parser.parse_args()
    init_fairseq2()
    if args.config_path is not None:
        cfg = get_slm_config(args.config_path, extracted_feature_folder=args.feature_dir)
        slm_eval_models(cfg.cfg_eval)
        return
    if any(value is None for value in (args.config_file, args.checkpoint, args.pairs_dir, args.out)):
        parser.error("--config-file, --checkpoint, --pairs-dir, and --out are required for recipe evaluation")
    import yaml

    from gslm2.lm.recipes.train.config import load_unit_lm_train_config

    with args.config_file.open() as handle:
        recipe = load_unit_lm_train_config(yaml.safe_load(handle))
    checkpoint = get_single_ckpt_path(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
    if recipe.model.vocab_size is None:
        parser.error("model.vocab_size must be set in the recipe config")
    try:
        slm_eval_model(
            checkpoint,
            args.pairs_dir,
            args.benchmarks,
            args.out,
            max_pairs=args.max_pairs,
            batch_size=args.batch_size,
            vocab_size=recipe.model.vocab_size,
            offset=recipe.dataset.unit_offset,
            model_preset=recipe.model_preset,
            seed=recipe.common.seed,
            distill_temperature=recipe.distill_temperature,
            depformer_generation_order=recipe.depformer_generation_order,
            model_config_overrides=parse_model_overrides(recipe.model.config_overrides),
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
