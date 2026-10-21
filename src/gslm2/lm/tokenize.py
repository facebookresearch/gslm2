# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import torch
import torch.distributed as dist
from fairseq2 import init_fairseq2
from fairseq2.data.data_pipeline import (
    DataPipeline,
    read_sequence,
)
from fairseq2.data.text import read_text
from fairseq2.device import CPU, Device
from fairseq2.gang import FakeGang, Gang, ProcessGroupGang
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from gslm2.shared.timing import timing

if TYPE_CHECKING:
    from collections.abc import Callable

    from fairseq2.data import DataPipelineBuilder
    from spacy.lang.en import English

    from gslm2.lm.config import LMPreparationConfig
    from gslm2.shared.distributed import DistributedContext


logger = logging.getLogger(__name__)


SLM21_ROOT = Path("/datasets/metaspeechssl/shared/data/zr-data/datasets/sLM21-dataset")


def _gold_text_lookup(gold_rel: str, text_column: str, *, lower: bool = False) -> Callable[[str], str]:
    """Map an sLM21 stimulus file stem to its transcript, from one split's ``gold.csv``."""
    by_filename: dict[str, str] | None = None

    def lookup(filename: str) -> str:
        nonlocal by_filename
        if by_filename is None:
            gold = pd.read_csv(SLM21_ROOT / gold_rel)
            by_filename = dict(zip(gold["filename"], gold[text_column], strict=False))
        value = by_filename.get(filename, "")
        return value.lower() if lower else value

    return lookup


def _get_lexical_text_lookup(split: str = "test") -> Callable[[str], str]:
    return _gold_text_lookup(f"lexical/{split}/gold.csv", "word")


def _get_syntactic_text_lookup(split: str = "test") -> Callable[[str], str]:
    return _gold_text_lookup(f"syntactic/{split}/gold.csv", "transcription", lower=True)


@dataclass
class TestSetConfig:
    sentencize: bool = True
    lookup_fn: Callable[[str], str] | None = None


TESTSET2CONFIG = {
    "lexical_dev": TestSetConfig(lookup_fn=_get_lexical_text_lookup("dev")),
    "lexical_test": TestSetConfig(lookup_fn=_get_lexical_text_lookup()),
    "syntactic_dev": TestSetConfig(lookup_fn=_get_syntactic_text_lookup("dev")),
    "syntactic_test": TestSetConfig(lookup_fn=_get_syntactic_text_lookup()),
    # Story-cloze sets carry each stimulus in its own .txt, so they need no gold lookup.
    "tSC_dev": TestSetConfig(),
    "tSC_dev_expanded": TestSetConfig(),
    "tSC": TestSetConfig(),
    "sSC": TestSetConfig(),
}


def clean_header(text: str) -> str:
    substring = "START OF THE PROJECT GUTENBERG"
    substring_pos = text.find(substring)
    if substring_pos == -1:
        substring = "START OF THIS PROJECT GUTENBERG"
        substring_pos = text.find(substring)

    if substring_pos == -1:
        return text

    return text[substring_pos + len(substring) :]


def clean_footer(text: str) -> str:
    substring = "END OF THE PROJECT GUTENBERG"
    substring_pos = text.find(substring)
    if substring_pos == -1:
        substring = "END OF THIS PROJECT GUTENBERG"
        substring_pos = text.find(substring)

    if substring_pos == -1:
        return text

    return text[:substring_pos]


def preprocess_text(text: str) -> str:
    text = clean_header(text)
    return clean_footer(text)


def _tokenize_raw_text(  # noqa: PLR0913
    raw_text: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    remove_license: bool,
    sentencize: bool,
    add_special_tokens: bool,
    sentencizer: object | None,
) -> list[int]:
    text = raw_text

    # Possibly apply some filtering on the raw text
    if remove_license:
        text = preprocess_text(text)

    # Sentencize
    if sentencize and sentencizer is not None:
        sentence_splitter = cast("English", sentencizer)
        doc = sentence_splitter(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        sentences = [text]

    # Batch call the tokenizer
    if sentences and sentences[0]:
        encodings = tokenizer(
            sentences,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        # flatten the tokenized sentences
        token_lists = cast("list[list[int]]", encodings)
        return [token for token_list in token_lists for token in token_list]

    return []


def create_pipeline_for_pairs(  # noqa: PLR0913
    pairs_jsonl: Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    remove_license: bool = True,
    sentencize: bool = True,
    add_special_tokens: bool = True,
    sentencizer: object | None = None,
    lookup_fn: Callable[[str], str] | None = None,
) -> DataPipeline:
    # no gang here since these files are typically small
    # An example of line:
    # {"positive": {"units": "2 34313...", "file_name": "/path/to/correct.txt"},
    #  "negative": {"units": "2 34313...", "file_name": "/path/to/incorrect.txt"}}
    builder = read_text(pairs_jsonl)

    def parse_line(line: str) -> dict[str, Any]:
        data = json.loads(line)
        # read the text in the file_name and tokenize it
        path_positive = Path(data["positive"]["file_name"]).resolve()
        if lookup_fn is not None:
            text_positive = lookup_fn(path_positive.stem)
        else:
            with path_positive.open(encoding="utf-8") as f:
                text_positive = f.read().strip()
        tokens_positive = _tokenize_raw_text(
            text_positive,
            tokenizer,
            remove_license=remove_license,
            sentencize=sentencize,
            add_special_tokens=add_special_tokens,
            sentencizer=sentencizer,
        )

        path_negative = Path(data["negative"]["file_name"]).resolve()
        if lookup_fn is not None:
            text_negative = lookup_fn(path_negative.stem)
        else:
            with path_negative.open(encoding="utf-8") as f:
                text_negative = f.read().strip()
        tokens_negative = _tokenize_raw_text(
            text_negative,
            tokenizer,
            remove_license=remove_license,
            sentencize=sentencize,
            add_special_tokens=add_special_tokens,
            sentencizer=sentencizer,
        )
        return {
            "positive": {
                "units": " ".join(map(str, tokens_positive)),
                "file_name": str(path_positive),
                "text": text_positive,
            },
            "negative": {
                "units": " ".join(map(str, tokens_negative)),
                "file_name": str(path_negative),
                "text": text_negative,
            },
        }

    builder.map(parse_line, num_parallel_calls=10)

    return builder.and_return()


def create_pipeline(  # noqa: PLR0913
    manifest: Path,
    gang: Gang,
    tokenizer: PreTrainedTokenizerBase,
    *,
    remove_license: bool = True,
    sentencize: bool = True,
    add_special_tokens: bool = True,
    sentencizer: object | None = None,
) -> DataPipeline:
    text_files = list(pd.read_csv(manifest)["file"])

    seed = 0

    def get_fileset_builder(text_files: list[str]) -> DataPipelineBuilder:
        builder = read_sequence(text_files)
        # Shuffle the files (to avoid any ordering artifact)
        builder.shuffle(shuffle_window=0, seed=seed)
        # Shard the list of files among ranks.
        builder.shard(gang.rank, gang.size, allow_uneven=True)
        return builder

    builder = get_fileset_builder(text_files)

    # Preprocess text files
    def read_file(text_file: str) -> str:
        with Path(text_file).open(encoding="utf-8") as f:
            return f.read()

    def tokenize_file(text_file: str) -> list[int]:
        text = read_file(text_file)
        return _tokenize_raw_text(
            text,
            tokenizer,
            remove_license=remove_license,
            sentencize=sentencize,
            add_special_tokens=add_special_tokens,
            sentencizer=sentencizer,
        )

    builder.map(tokenize_file, num_parallel_calls=10)

    with_file = DataPipeline.zip(
        [builder.and_return(), get_fileset_builder(text_files).and_return()], names=["tokens", "file"]
    )

    return with_file.and_return()


def preprocess_manifest(  # noqa: PLR0913
    manifest: Path,
    output_folder: Path,
    gang: Gang,
    tokenizer: PreTrainedTokenizerBase,
    *,
    remove_license: bool = True,
    sentencize: bool = True,
    add_special_tokens: bool = True,
    sentencizer: object | None = None,
) -> None:
    pipeline = create_pipeline(
        manifest,
        gang,
        tokenizer,
        remove_license=remove_license,
        sentencize=sentencize,
        add_special_tokens=add_special_tokens,
        sentencizer=sentencizer,
    )
    for tokenized_file in pipeline:
        token_list = cast("list[int]", tokenized_file["tokens"])
        file_path = cast("str", tokenized_file["file"])
        # Build some kind of human-readable unique filename for the .txt
        manifest_stem = manifest.stem
        file_path_obj = Path(file_path)
        parts = file_path_obj.parts
        try:
            idx = parts.index(manifest_stem)
            filename = "_".join(parts[idx + 1 :])
        except ValueError:
            filename = file_path_obj.name
        # write the tokens to a file
        output_file = output_folder / filename
        output_folder.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(" ".join(map(str, token_list)))


@timing
def save_tokens(
    cfg: LMPreparationConfig,
    context: DistributedContext,
) -> None:
    logger.info("Distributed context %s", context)
    local_rank = context.local_rank
    world_size = context.world_size

    # Init fairseq2
    init_fairseq2()

    # we won't use the GPU, we just do this for gangs
    device = Device(f"cuda:{local_rank}") if torch.cuda.is_available() else CPU

    if world_size > 1 and dist.group.WORLD is not None:
        gang = ProcessGroupGang(dist.group.WORLD, device)
    else:
        gang = FakeGang(device)

    sentencizer = None
    if cfg.text_sentencize:
        from spacy.lang.en import English

        sentencizer = English()
        sentencizer.add_pipe("sentencizer")
        sentencizer.max_length = 1000000000  # or even higher

    tokenizer: PreTrainedTokenizerBase
    if cfg.text_tokenizer_path is not None:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=cfg.text_tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    for split in ["train", "valid"]:
        for manifest_ in getattr(cfg, f"manifests_{split}", []):
            manifest = Path(manifest_).resolve()
            logger.info("Extracting tokens for %s manifest %s", split, manifest)
            preprocess_manifest(
                manifest,
                cfg.output_folder / split,
                gang,
                tokenizer,
                remove_license=cfg.text_remove_license,
                sentencize=cfg.text_sentencize,
                add_special_tokens=cfg.text_add_special_tokens,
                sentencizer=sentencizer,
            )

    if context.is_leader:
        for test_jsonl_ in cfg.manifests_test:
            test_jsonl = Path(test_jsonl_).resolve()
            logger.info("Extracting tokens for test manifest %s", test_jsonl)
            test_set_config = TESTSET2CONFIG.get(test_jsonl.stem, TestSetConfig())
            target_jsonl = cfg.output_folder / "test" / "pairs" / test_jsonl.name
            target_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with target_jsonl.open("w", encoding="utf-8") as fout:
                for line in create_pipeline_for_pairs(
                    test_jsonl,
                    tokenizer,
                    remove_license=cfg.text_remove_license,
                    sentencize=cfg.text_sentencize and test_set_config.sentencize,
                    add_special_tokens=cfg.text_add_special_tokens,
                    sentencizer=sentencizer,
                    lookup_fn=test_set_config.lookup_fn,
                ):
                    fout.write(json.dumps(line) + "\n")
