# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical text normalization shared by speech and text candidates."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import import_module
from typing import Protocol, cast

_TEEN_LIMIT = 20
_TENS_LIMIT = 100
_HUNDRED = 100
_THOUSAND = 1_000
_MILLION = 1_000_000
_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ORDINAL_ONES = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|n[d]|rd|th)\b")
_INT_RE = re.compile(r"\b(\d+)\b")


class _Normalizer(Protocol):
    def __call__(self, text: str) -> str: ...


def _int_to_words(number: int) -> str:
    if number < _TEEN_LIMIT:
        return _ONES[number]
    if number < _TENS_LIMIT:
        tens, ones = divmod(number, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_ONES[ones]}"
    for limit, divisor, name in ((_THOUSAND, _HUNDRED, "hundred"), (_MILLION, _THOUSAND, "thousand")):
        if number < limit:
            head_count, rest = divmod(number, divisor)
            head = f"{_int_to_words(head_count)} {name}"
            return head if rest == 0 else f"{head} {_int_to_words(rest)}"
    millions, rest = divmod(number, _MILLION)
    head = f"{_int_to_words(millions)} million"
    return head if rest == 0 else f"{head} {_int_to_words(rest)}"


def _to_ordinal_words(number: int) -> str:
    words = _int_to_words(number)
    head, _, last = words.rpartition(" ")
    if last in _ORDINAL_ONES:
        last = _ORDINAL_ONES[last]
    elif last.endswith("y"):
        last = f"{last[:-1]}ieth"
    else:
        last = f"{last}th"
    return f"{head} {last}".strip()


@lru_cache(maxsize=1)
def _english_normalizer() -> _Normalizer:
    normalizers = import_module("whisper.normalizers")
    normalizer_class = cast("type[_Normalizer]", normalizers.EnglishTextNormalizer)
    return normalizer_class()


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, expand contractions, and spell out numerals."""
    if not text.strip():
        return ""
    normalized = _english_normalizer()(text)
    normalized = _ORDINAL_RE.sub(lambda match: _to_ordinal_words(int(match.group(1))), normalized)
    normalized = _INT_RE.sub(lambda match: _int_to_words(int(match.group(1))), normalized)
    return " ".join(normalized.split())
