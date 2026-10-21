import argparse
import logging
from pathlib import Path

from gslm2.lm.config import get_slm_config, layer_num_str
from gslm2.lm.recipes.train.recipe import UnitLMTrainRecipe
from gslm2.lm.runtime.profiling import cuda_memory_snapshot
from gslm2.lm.runtime.recipe.run import train as train_slm

logger = logging.getLogger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        help="Path to the pre-extracted feature directory.",
        default=None,
    )
    parser.add_argument(
        "--slurm_job_name",
        type=str,
        help="Override the SLURM job name (e.g. SLM_7399879) to resume a previous run.",
        default=None,
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    extracted_feature_dir = None
    if args.feature_dir:
        extracted_feature_dir = Path(args.feature_dir)
    cfg_slm = get_slm_config(
        config_path, extracted_feature_folder=extracted_feature_dir, slurm_job_name=args.slurm_job_name
    )

    logger.info("Training the Spoken Language Model")
    layer_folder = (
        cfg_slm.cfg_feat_ext.output_folder / layer_num_str(cfg_slm.cfg_feat_ext.effective_output_layers) / "ckpt"
    )

    recipe = UnitLMTrainRecipe()

    profile_enabled = cfg_slm.cfg_trainer_fs2.common.profilers.torch.enabled
    with cuda_memory_snapshot(layer_folder, enabled=profile_enabled):
        train_slm(recipe, cfg_slm.cfg_trainer_fs2, output_dir=layer_folder)


if __name__ == "__main__":
    main()
