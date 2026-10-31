# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal unit-conditioned HiFi-GAN models and training utilities."""

from gslm2.vocoder.model import SAMPLE_RATE, UnitDiscriminator, UnitVocoder

__all__ = ["SAMPLE_RATE", "UnitDiscriminator", "UnitVocoder"]
