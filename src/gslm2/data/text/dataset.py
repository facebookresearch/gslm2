"""Text dataset and dataloader factory for causal-LM training (``.txt`` / ``.json`` manifests)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from gslm2.data.common.weighted_sampling import WeightedDistributedBatchSampler
from gslm2.data.registry import resolve_manifest_uri
from gslm2.data.text.collators import CausalCollator, PackedCausalCollator, TextSample

if TYPE_CHECKING:
    from gslm2.config.data import TextDataConfig

logger = logging.getLogger(__name__)


def _load_tokenizer(name_or_path: str):  # noqa: ANN202
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase:
    """Load a HF tokenizer, defaulting a missing pad token to EOS."""
    return _load_tokenizer(name_or_path)


def _read_manifest_texts(manifest_path: str, text_key: str) -> list[str]:
    p = Path(manifest_path)
    if p.suffix == ".txt":
        with p.open("r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]
    if p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"Expected a JSON list in {manifest_path}, got {type(entries).__name__}")
        return [e[text_key] for e in entries if e.get(text_key)]
    raise ValueError(f"Unsupported text manifest format: {p.suffix} ({manifest_path})")


class TextDataset(Dataset):
    """Dataset that loads text samples from ``.txt`` / ``.json`` manifest files."""

    def __init__(self, data_cfg: TextDataConfig) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.dataset_sizes: list[int] = []
        for spec in data_cfg.manifests:
            manifest_path = resolve_manifest_uri(spec)
            items = _read_manifest_texts(manifest_path, data_cfg.text_key)
            self.texts.extend(items)
            self.dataset_sizes.append(len(items))
            logger.info("Loaded %d text samples from %s", len(items), manifest_path)

        self.max_length = data_cfg.max_length
        self.add_special_tokens = data_cfg.add_special_tokens
        self.task = data_cfg.task
        self.tokenizer = load_tokenizer(data_cfg.tokenizer)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> TextSample:
        encoding = self.tokenizer(
            self.texts[index],
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return TextSample(input_ids=encoding["input_ids"][0].to(torch.long))


def build_text_dataloader(data_cfg: TextDataConfig) -> DataLoader:
    """Build a causal-LM text DataLoader."""
    dataset = TextDataset(data_cfg)
    pad_id = dataset.tokenizer.pad_token_id

    if data_cfg.task != "causal":
        raise ValueError(f"Unsupported TextDataConfig.task: {data_cfg.task!r}")
    if data_cfg.packing:
        collate_fn = PackedCausalCollator(pad_token_id=pad_id, max_length=data_cfg.max_length)
    else:
        collate_fn = CausalCollator(pad_token_id=pad_id)

    sample_weights: list[float] = []
    for i, (w, size) in enumerate(zip(data_cfg.manifests_weights, dataset.dataset_sizes, strict=True)):
        if size == 0:
            msg = f"manifest {i} ({data_cfg.manifests[i]}) yielded 0 samples — check the path/filter"
            raise ValueError(msg)
        sample_weights.extend([w / size] * size)
    batch_sampler = WeightedDistributedBatchSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        batch_size=data_cfg.batch_size,
        seed=data_cfg.random_seed,
        drop_last=data_cfg.drop_last,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=data_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=data_cfg.persistent_workers and data_cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(
            data_cfg.random_seed + (dist.get_rank() if dist.is_initialized() else 0)
        ),
    )
