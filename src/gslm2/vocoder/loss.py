# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""HiFi-GAN adversarial, feature-matching, and mel losses."""

import torch
from torch import Tensor

from gslm2.vocoder.model import UnitDiscriminatorOutput

type LossParts = dict[str, Tensor]

LAMBDA_MEL = 45.0
LAMBDA_FEATURE_MATCHING = 2.0


def feature_matching_loss(real_features: tuple[Tensor, ...], generated_features: tuple[Tensor, ...]) -> Tensor:
    """Sum mean absolute errors over discriminator feature maps."""
    return torch.stack(
        [(generated - real).abs().mean() for real, generated in zip(real_features, generated_features, strict=True)]
    ).sum()


def gan_generator_loss(generated_scores: tuple[Tensor, ...]) -> Tensor:
    """Compute least-squares GAN generator loss."""
    return torch.stack([(score - 1).square().mean() for score in generated_scores]).sum()


def gan_discriminator_loss(real_scores: tuple[Tensor, ...], generated_scores: tuple[Tensor, ...]) -> Tensor:
    """Compute least-squares GAN discriminator loss."""
    real_losses = [(score - 1).square().mean() for score in real_scores]
    generated_losses = [score.square().mean() for score in generated_scores]
    return torch.stack(real_losses + generated_losses).sum()


def discriminator_loss(
    real: UnitDiscriminatorOutput,
    generated: UnitDiscriminatorOutput,
) -> tuple[Tensor, LossParts]:
    """Combine multi-period and multi-scale discriminator losses."""
    mpd = gan_discriminator_loss(real.mpd, generated.mpd)
    msd = gan_discriminator_loss(real.msd, generated.msd)
    return mpd + msd, {"mpd": mpd, "msd": msd}


def generator_loss(
    real: UnitDiscriminatorOutput,
    generated: UnitDiscriminatorOutput,
) -> tuple[Tensor, LossParts]:
    """Compute the canonical HiFi-GAN generator objective."""
    mel = (real.log_mel_spectrogram - generated.log_mel_spectrogram).abs().mean()
    fm_mpd = feature_matching_loss(real.mpd_features, generated.mpd_features)
    fm_msd = feature_matching_loss(real.msd_features, generated.msd_features)
    mpd = gan_generator_loss(generated.mpd)
    msd = gan_generator_loss(generated.msd)
    loss = LAMBDA_MEL * mel + LAMBDA_FEATURE_MATCHING * (fm_mpd + fm_msd) + mpd + msd
    return loss, {"mel": mel, "fm_mpd": fm_mpd, "fm_msd": fm_msd, "mpd": mpd, "msd": msd}
