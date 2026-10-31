# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path

import torch.distributed as dist

from gslm2.lm.config import LMPreparationConfig, get_preparation_config
from gslm2.lm.eval.create_pairs import create_paired_file
from gslm2.shared.distributed import DistributedContext, init_distributed_context

logger = logging.getLogger()


def run_feature_extraction(cfg_feat_ext: LMPreparationConfig, context: DistributedContext) -> None:
    if cfg_feat_ext.model_type == "text":
        from gslm2.lm.tokenize import save_tokens

        save_tokens(cfg_feat_ext, context)
    else:
        from gslm2.encoder.units.extract import run_feature_extraction_speech

        run_feature_extraction_speech(cfg_feat_ext, context)
        if context.is_leader:
            for layer in cfg_feat_ext.effective_output_layers:
                for manifest in cfg_feat_ext.manifests_test:
                    create_paired_file(
                        cfg_feat_ext.output_folder / str(layer) / "test" / f"{manifest.stem}.jsonl",
                        slm21_root=cfg_feat_ext.slm21_root,
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    cfg = get_preparation_config(config_path)
    context = init_distributed_context(cfg.distributed_port)
    try:
        run_feature_extraction(cfg, context)
        if context.is_distributed:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
