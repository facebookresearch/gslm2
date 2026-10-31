# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train and run duration predictors for deduplicated speech units."""

from gslm2.duration.config import DurationPredictorConfig
from gslm2.duration.model import DurationPredictor

__all__ = ["DurationPredictor", "DurationPredictorConfig"]
