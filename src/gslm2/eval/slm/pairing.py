# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared SLM benchmark metadata + pairing predicates.

Used by both create_pairs.py (precomputed-codes path, pandas, JSONL output)
and scripts/slm/create_slm_pair_manifests.py (on-the-fly path, polars, CSV
output).
"""

from __future__ import annotations

from typing import Final

# ---- Benchmark categorization -----------------------------------------------

STORYCLOZE_BENCHMARKS: Final[frozenset[str]] = frozenset({"sSC", "tSC", "tSC_dev", "tSC_dev_expanded"})

ZR21_BENCHMARKS: Final[tuple[str, ...]] = (
    "lexical_dev",
    "lexical_dev_invocab",
    "lexical_dev_oov",
    "lexical_test",
    "lexical_test_invocab",
    "lexical_test_oov",
    "syntactic_dev",
    "syntactic_test",
)

# Multilingual benchmarks: correct/incorrect pairing encoded in filename
_ML_VOICES: Final[tuple[str, ...]] = (
    "fr-FR-Chirp3-HD-Aoede",
    "fr-FR-Chirp3-HD-Charon",
    "fr-FR-Chirp3-HD-Fenrir",
    "fr-FR-Chirp3-HD-Kore",
    "fr-FR-Chirp3-HD-Puck",
    "fr-FR-Chirp3-HD-Zephyr",
    "de-DE-Chirp3-HD-Aoede",
    "de-DE-Chirp3-HD-Charon",
    "de-DE-Chirp3-HD-Fenrir",
    "de-DE-Chirp3-HD-Kore",
    "de-DE-Chirp3-HD-Puck",
    "de-DE-Chirp3-HD-Zephyr",
)
ML_TSC_BENCHMARKS: Final[frozenset[str]] = frozenset(f"tsc_{v}" for v in _ML_VOICES)
ML_BLIMP_BENCHMARKS: Final[frozenset[str]] = frozenset(f"blimp_{v}" for v in _ML_VOICES)

# ---- ZR21 lookup tables -----------------------------------------------------

# Group-by columns for pairing within gold.csv. Syntactic adds "subtype".
ZR21_GROUP_COLS: Final[dict[str, tuple[str, ...]]] = {
    b: ("id", "voice", "subtype") if "syntactic" in b else ("id", "voice") for b in ZR21_BENCHMARKS
}

# gold.csv subpath for sLM21-dataset/ root (precomputed: hardcoded; on-the-fly: --audio-root flag)
ZR21_GOLD_SUBPATH: Final[dict[str, str]] = {
    b: f"{'syntactic' if 'syntactic' in b else 'lexical'}/{'dev' if 'dev' in b else 'test'}" for b in ZR21_BENCHMARKS
}

# StoryCloze fileid stem format: "{story_id}_{uuid}_{correct|incorrect}", or
# "{story_id}_{uuid}_seed{n}_{correct|incorrect}" for the expanded (dev) manifests.
# UUIDs contain hyphens not underscores, so split("_") is unambiguous.


def is_storycloze_correct(stem: str) -> bool:
    """True iff the StoryCloze fileid stem ends in '_correct'."""
    return stem.rsplit("_", 1)[-1] == "correct"


def storycloze_pair_id(stem: str, *, include_seed: bool = False) -> str:
    """Extract the key that joins a correct/incorrect StoryCloze pair.

    For expanded manifests the seed token (``seed{n}``) is folded into the key so seed-varied
    stories of the same uuid pair correctly.
    """
    parts = stem.split("_")
    pair_id = parts[1]
    if include_seed:
        pair_id = f"{pair_id}_{parts[2]}"
    return pair_id
