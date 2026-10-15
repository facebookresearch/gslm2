import argparse
import logging
from pathlib import Path

from gslm2.eval.slm.config import FeatExtractConfig, get_slm_config
from gslm2.eval.slm.distributed import DistributedContext, init_distributed_context
from gslm2.eval.slm.extract_features import run_feature_extraction_speech
from gslm2.eval.slm.extract_text_tokens import save_tokens

logger = logging.getLogger()


def run_feature_extraction(cfg_feat_ext: FeatExtractConfig, context: DistributedContext) -> None:
    if cfg_feat_ext.model_type == "text":
        save_tokens(cfg_feat_ext, context)
    else:
        run_feature_extraction_speech(cfg_feat_ext, context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    cfg_slm = get_slm_config(config_path)

    context = init_distributed_context(cfg_slm.cfg_feat_ext.distributed_port)
    run_feature_extraction(cfg_slm.cfg_feat_ext, context)
