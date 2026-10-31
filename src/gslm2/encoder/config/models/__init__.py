# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Encoder model configuration dispatch."""

from gslm2.encoder.config.common import ModelType
from gslm2.encoder.config.models.base import BaseModelConfig
from gslm2.encoder.config.models.speech import SpidRConfig


def model_type_to_config_type(model_type: ModelType) -> type[BaseModelConfig]:
    if model_type != "spidr":
        raise ValueError(f"Unknown encoder model type: {model_type}")
    return SpidRConfig
