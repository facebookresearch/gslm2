import argparse
import logging
import os
from pathlib import Path

import torch.distributed as dist

from gslm2.eval.slm.config import SLMConfig, get_slm_config, layer_num_str
from gslm2.eval.slm.distributed import DistributedContext, init_distributed_context
from gslm2.eval.slm.profiling import cuda_memory_snapshot
from gslm2.eval.slm.slm_eval import slm_eval_models
from gslm2.eval.slm.slm_feat_ext import run_feature_extraction
from gslm2.fs2.lib.recipe.run import train as train_slm
from gslm2.fs2.unit_lm.recipes.train.dataset_speech_manifest import ManifestUnitDatasetConfig
from gslm2.fs2.unit_lm.recipes.train.dataset_text import TextUnitDatasetConfig
from gslm2.fs2.unit_lm.recipes.train.recipe import UnitLMTrainRecipe
from gslm2.fs2.unit_lm.recipes.train.storage_s3_eoz import resolve_manifest_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_config_run_pipeline(  # noqa: PLR0913
    config_path: Path,
    common_folder: Path,
    output_results_file: Path,
    config_updates: dict,
    extracted_feature_folder: Path | None = None,
    slurm_job_name: str | None = None,
) -> None:
    config = get_slm_config(
        path=config_path,
        common_folder=common_folder,
        output_results_file=output_results_file,
        dict_update=config_updates,
        extracted_feature_folder=extracted_feature_folder,
        slurm_job_name=slurm_job_name,
    )
    run_slm_pipeline(config, skip_feature_extraction=extracted_feature_folder is not None)


def reads_published_tokens(cfg_slm: SLMConfig) -> bool:
    return isinstance(
        cfg_slm.cfg_trainer_fs2.dataset.config_overrides,
        ManifestUnitDatasetConfig | TextUnitDatasetConfig,
    )


def verify_published_splits(cfg_slm: SLMConfig, context: DistributedContext) -> None:
    """Verify each split the config names exists under the published store."""
    root = resolve_manifest_root(cfg_slm.cfg_trainer_fs2.dataset.config_overrides.path)
    dataset = cfg_slm.cfg_trainer_fs2.dataset
    splits = [dataset.train_split]
    if dataset.valid_split:
        splits.append(dataset.valid_split)
    splits.extend(dataset.aux_valid_splits)
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split '{split}' is not published: missing {split_dir}")
    if context.is_leader:
        logger.info("Verified published splits %s under %s", splits, root)


def verify_reused_features(cfg_slm: SLMConfig, context: DistributedContext) -> None:
    """When reusing pre-extracted tokens, verify the train/valid token files exist
    and that shard counts are divisible by the world size (required by the reader)."""
    feature_dir = Path(cfg_slm.cfg_trainer_fs2.dataset.config_overrides.path)
    dataset = cfg_slm.cfg_trainer_fs2.dataset
    splits = [dataset.train_split]
    if dataset.valid_split:
        splits.append(dataset.valid_split)
    for split in splits:
        split_dir = feature_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Cannot reuse features: missing split dir '{split}' at {split_dir}")
        n = len(sorted(split_dir.glob("*.txt")))
        if n == 0:
            raise FileNotFoundError(f"Cannot reuse features: no '.txt' token files in {split_dir}")
        if context.world_size > 1 and n % context.world_size != 0:
            raise ValueError(
                f"Token shard count in {split_dir} ({n}) is not divisible by world_size "
                f"({context.world_size}). Use a node/GPU count that divides {n}."
            )
    if context.is_leader:
        logger.info("Verified reused features at %s", feature_dir)


def run_slm_pipeline(cfg_slm: SLMConfig, *, skip_feature_extraction: bool = False) -> None:  # noqa: C901, PLR0912
    if cfg_slm.cfg_trainer_fs2.is_train_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Step 1: Feature extraction
    context = init_distributed_context(cfg_slm.cfg_feat_ext.distributed_port)
    if context.is_leader:
        logger.info("Distributed context %s", context)

    if reads_published_tokens(cfg_slm):
        if context.is_leader:
            logger.info("=== Skipping Step 1: reading published token manifests ===")
        verify_published_splits(cfg_slm, context)
        cfg_slm.cfg_eval.delete_data_model_after_eval = False
    elif skip_feature_extraction:
        if context.is_leader:
            logger.info("=== Skipping Step 1: reusing pre-extracted features ===")
        verify_reused_features(cfg_slm, context)
        # CRITICAL: the reused feature folder is external.
        # Step-3 eval deletes `feature_dir_override.parent` when this is True. Force it off when reusing features.
        if cfg_slm.cfg_eval.delete_data_model_after_eval:
            if context.is_leader:
                logger.warning("Disabling 'delete_data_model_after_eval' because features are being reused.")
            cfg_slm.cfg_eval.delete_data_model_after_eval = False
    else:
        if context.is_leader:
            logger.info("=== Starting Step 1: Feature Extraction ===")
        run_feature_extraction(cfg_slm.cfg_feat_ext, context)
        if context.is_leader:
            logger.info("Step 1: Feature extraction completed successfully")

    if context.world_size > 1:
        dist.barrier()

    # Step 2: SLM training
    if context.is_leader:
        logger.info("=== Starting Step 2: SLM Training ===")

    layer_folder = (
        cfg_slm.cfg_feat_ext.output_folder / layer_num_str(cfg_slm.cfg_feat_ext.effective_output_layers) / "ckpt"
    )

    recipe = UnitLMTrainRecipe()
    profile_enabled = cfg_slm.cfg_trainer_fs2.common.profilers.torch.enabled
    with cuda_memory_snapshot(layer_folder, enabled=profile_enabled):
        train_slm(recipe, cfg_slm.cfg_trainer_fs2, output_dir=layer_folder)

    if context.is_leader:
        logger.info("Step 2: SLM training completed successfully")

    if context.world_size > 1:
        dist.barrier()
    else:
        logger.info("End of Step 2: dist not initialized ??")

    # Step 3: SLM evaluation
    if context.is_leader:
        logger.info("=== Starting Step 3: SLM Evaluation ===")

    slm_eval_models(cfg_slm.cfg_eval)

    if context.is_leader:
        logger.info("=== SLM Pipeline completed successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM Pipeline: Feature extraction, training, and evaluation")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument(
        "--feature_dir", type=str, default=None, help="Override feature directory (for starting from step 2 or 3)"
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    extracted_feature_dir = Path(args.feature_dir) if args.feature_dir else None
    logger.info("Config: %s", config_path)
    if extracted_feature_dir:
        logger.info("Feature dir override: %s", extracted_feature_dir)

    # Get configurations
    cfg_slm = get_slm_config(config_path, extracted_feature_folder=extracted_feature_dir)

    run_slm_pipeline(cfg_slm, skip_feature_extraction=extracted_feature_dir is not None)
