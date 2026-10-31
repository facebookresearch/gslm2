# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Semantic-preserving normalization for LLM judging."""

from __future__ import annotations

import re
import unicodedata

_APOSTROPHE_TRANSLATION = str.maketrans({"`": "'", "\u2018": "'", "\u2019": "'", "\u02bc": "'"})
_NON_SEMANTIC_PUNCTUATION = re.compile(r"[^\w\s.!?'$\u00a3\u20ac%]+")
_REPEATED_TERMINATOR = re.compile(r"([.!?])(?:\s*[.!?])+")
_SPACE_BEFORE_TERMINATOR = re.compile(r"\s+([.!?])")


def normalize_for_judge(text: str) -> str:
    """Lowercase text while retaining apostrophes and sentence boundaries."""
    if not text.strip():
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION).lower()
    normalized = normalized.replace("&", " and ").replace("_", " ")
    normalized = _NON_SEMANTIC_PUNCTUATION.sub(" ", normalized)
    normalized = _REPEATED_TERMINATOR.sub(r"\1", normalized)
    normalized = _SPACE_BEFORE_TERMINATOR.sub(r"\1", normalized)
    return " ".join(normalized.split())
