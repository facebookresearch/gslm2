# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Continuation repetition metrics and S12 aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class RepetitionRates:
    """Percentages of positions occupied by repeated unigrams and bigrams."""

    unigram: float
    bigram: float


def repetition_rates(normalized_text: str, window_words: int) -> RepetitionRates:
    """Measure repeated unigram and adjacent-bigram positions in the first W words."""
    if window_words < 1:
        raise ValueError("window_words must be positive")
    words = normalized_text.split()[:window_words]
    unigram_counts = Counter(words)
    repeated_unigrams = sum(unigram_counts[word] > 1 for word in words)
    unigram = 100.0 * repeated_unigrams / len(words) if words else 0.0

    bigrams = list(pairwise(words))
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(bigram_counts[bigram] > 1 for bigram in bigrams)
    bigram = 100.0 * repeated_bigrams / len(bigrams) if bigrams else 0.0
    return RepetitionRates(unigram=unigram, bigram=bigram)


def s12(candidate_rates: Sequence[RepetitionRates], reference_rates: Sequence[RepetitionRates]) -> float:
    """Compute run-level S12 from mean candidate and reference repetition rates."""
    if not candidate_rates or not reference_rates:
        raise ValueError("S12 requires candidate and reference repetition rates")
    candidate_r1 = sum(rate.unigram for rate in candidate_rates) / len(candidate_rates)
    candidate_r2 = sum(rate.bigram for rate in candidate_rates) / len(candidate_rates)
    reference_r1 = sum(rate.unigram for rate in reference_rates) / len(reference_rates)
    reference_r2 = sum(rate.bigram for rate in reference_rates) / len(reference_rates)
    if reference_r1 == 0.0 or reference_r2 == 0.0:
        raise ValueError("reference repetition rates must be non-zero to compute S12")
    return 0.5 * (candidate_r1 / reference_r1 + candidate_r2 / reference_r2)
