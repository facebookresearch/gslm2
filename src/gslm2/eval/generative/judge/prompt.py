# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The fixed v3.1-short continuation-judging prompt."""

from __future__ import annotations

import hashlib

_MODALITY_NOTE = (
    "The prompt and response use the same semantic normalization: lowercase text with "
    "apostrophes and sentence boundaries preserved. They may contain minor rendering or "
    "transcription artifacts. Judge the intended content when it remains understandable."
)

PROMPT_TEMPLATE = """\
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
Evaluate the passage below. <CONTINUATION_START> separates the given prompt from the generated
continuation. Score only the continuation, but read both sides together. The boundary may be inside
a sentence, name, quotation, or question, and the continuation may end at a generation limit.

{modality_note}

###Response to evaluate:
{prompt_transcript}<CONTINUATION_START>{continuation_transcript}

###Reference Answer (Score 5):
Any natural, coherent, non-degenerate continuation that fits the marked boundary. Multiple answers
are valid; it need not match a particular event, add detail, or finish the passage.

###Score Rubrics:
[Semantic continuation quality: Judge boundary fit, internal coherence, and non-degeneration from
the actual words. Missing boundary punctuation and newly introduced entities are not errors when
the connection is clear. Ordinary word reuse is cohesion; claim copying only for an actual repeated
multiword phrase. Forgive isolated understandable transcription artifacts and ignore only the final
unfinished fragment. Do not invent missing links or let a coherent opening excuse later unrelated,
contradictory, repetitive, or uninterpretable text. Choose the highest fully satisfied score. Give
at most two feedback sentences and 60 words, cite the strongest evidence, and end with [RESULT] N.]
Score 1: No connected meaning is recoverable, or most of the continuation is unrelated, contradictory, uninterpretable, copied, boilerplate, or looping.
Score 2: A local connection exists, but major or repeated failures make the overall meaning difficult to recover.
Score 3: The main meaning is recoverable and mostly connected, but a substantive defect or several local errors materially weaken it.
Score 4: The continuation fits and remains coherent, with only localized minor errors and no substantive defect.
Score 5: Every complete span fits, is coherent and natural, and has no identified semantic defect.

###Feedback:
"""

PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    PROMPT_TEMPLATE.replace("{modality_note}", _MODALITY_NOTE).encode()
).hexdigest()


def build_prompt(prompt_text: str, continuation_text: str) -> str:
    """Render the fixed judge prompt for one continuation."""
    return PROMPT_TEMPLATE.format(
        modality_note=_MODALITY_NOTE,
        prompt_transcript=prompt_text,
        continuation_transcript=continuation_text,
    )
