import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import replace
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
from gslm2.lm.models.opt import create_opt_model, get_opt_config_preset
from gslm2.lm.models.opt.config import MultiStreamOPTConfig
from gslm2.lm.runtime.utils.checkpoint import unify_fsdp_checkpoint
from gslm2.shared.distributed import DistributedContext
from gslm2.shared.utils.typing import PathLike

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout
)
logger = logging.getLogger(__name__)


def slm_eval_model(  # noqa: PLR0913, PLR0915
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
    gang = ProcessGroupGang(dist.group.WORLD, device) if world_size > 1 else FakeGang(device)

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
    opt_config = get_opt_config_preset(model_preset)
    if isinstance(opt_config, MultiStreamOPTConfig):
        assert isinstance(vocab_size, list)
        opt_config.vocab_size = [vs + offset for vs in vocab_size]
        opt_config.distill_temperature = distill_temperature
        opt_config.depformer_generation_order = depformer_generation_order
    else:
        assert isinstance(vocab_size, int)
        opt_config.vocab_size = vocab_size + offset
    if model_config_overrides:
        opt_config = replace(opt_config, **model_config_overrides)
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


def get_single_ckpt_path(  # noqa: C901
    step_dir: Path,
    dist_context: DistributedContext | None = None,
) -> Path:
    """Resolve the model checkpoint in a step dir.

    DDP writes one model file and is returned as-is; FSDP writes one sdp_* shard per
    rank and is unified first.
    """
    # Find all .pt files recursively
    all_pt_files = list(step_dir.rglob("*.pt"))

    # Filter to only include model-related .pt files:
    # 1. Files directly named "model.pt"
    # 2. Files in paths containing "model", essentially sdp_XX.pt sharded files
    model_pt_files = [
        pt_file
        for pt_file in all_pt_files
        if pt_file.name == "model.pt" or "model" in str(pt_file.relative_to(step_dir)).lower()
    ]

    if len(model_pt_files) == 0:
        raise ValueError(f"No model .pt files found in {step_dir}")

    if len(model_pt_files) == 1:
        return model_pt_files[0]

    # Check if there's exactly one "model.pt" file, and prioritize this one
    model_pt_candidates = [pt_file for pt_file in model_pt_files if pt_file.name == "model.pt"]
    if len(model_pt_candidates) == 1:
        return model_pt_candidates[0]
    if len(model_pt_candidates) > 1:
        raise ValueError(f"Multiple 'model.pt' files found in {step_dir}: {model_pt_candidates}")

    # Check for all sharded checkpoints
    if not all(pt_file.name.startswith("sdp_") for pt_file in model_pt_files):
        raise ValueError(f"Not all .pt files in {step_dir} start with 'sdp_' (sharded params): {model_pt_files}")

    # Sync here so that a single unification is performed, and all processes wait for it to complete
    if dist_context is not None:
        dist.barrier()

    # Unify the checkpoint
    unified_ckpt_dir = step_dir / "unified_ckpt"
    if dist_context is None or dist_context.rank == 0:
        logger.info("Unifying FSDP checkpoint in %s to %s ...", step_dir, unified_ckpt_dir)

        # Find the longest parent path ending with "model/"
        sharded_ckpt_dir = model_pt_files[0]
        while sharded_ckpt_dir != sharded_ckpt_dir.parent:
            if sharded_ckpt_dir.name == "model":
                break
            sharded_ckpt_dir = sharded_ckpt_dir.parent

        unify_fsdp_checkpoint(
            sharded_ckpt_dir=sharded_ckpt_dir,
            unified_ckpt_dir=unified_ckpt_dir,
            pp_size=1,
            tp_size=1,
        )

    if dist_context is not None:
        dist.barrier()

    # Unified checkpoints use the same layout DDP writes: pp_00/tp_00/sdp_00.pt.
    model_pt_file = unified_ckpt_dir / "pp_00" / "tp_00" / "sdp_00.pt"
    if not model_pt_file.exists():
        raise ValueError(
            f"FSDP-sharded checkpoint in {step_dir} could not be unified. "
            f"Expected unified file at {model_pt_file} but it was not found."
        )

    return model_pt_file


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
            model_config_overrides=recipe.model.config_overrides,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
