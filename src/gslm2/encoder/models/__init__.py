# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SpidR speech encoder and checkpoint loading."""

from gslm2.encoder.models.factory import build_model
from gslm2.encoder.models.spidr import SpidR

__all__ = ["SpidR", "build_model"]
