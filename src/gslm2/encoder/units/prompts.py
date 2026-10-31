# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Crop waveform prompts before bidirectional speech encoding."""

import json
import typing as tp
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from gslm2.encoder.data.collators import conv_length
from gslm2.shared.audio_io import normalize_waveform
from gslm2.shared.constants import AUDIO_SAMPLE_RATE


def minimum_input_samples(num_output_frames: int, conv_layers: list[tuple[int, int, int]]) -> int:
    """Return the shortest waveform producing the requested convolution frames."""
    if num_output_frames <= 0:
        raise ValueError(f"num_output_frames must be positive, got {num_output_frames}")
    num_samples = num_output_frames
    for _, kernel_size, stride in reversed(conv_layers):
        num_samples = (num_samples - 1) * stride + kernel_size
    return num_samples


class PromptCroppedDataset(Dataset):
    """Normalize audio prefixes and pad to the boundary's requested frame count."""

    def __init__(
        self,
        original_dataset: Dataset,
        boundary_manifest: Path,
        conv_layers: list[tuple[int, int, int]],
    ) -> None:
        self.original_dataset = original_dataset
        self.conv_layers = conv_layers
        self.boundaries: dict[str, tuple[float, int]] = {}
        with boundary_manifest.open(encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                file_id = str(record["fileid"])
                if file_id in self.boundaries:
                    raise ValueError(f"duplicate prompt boundary for {file_id!r} at {boundary_manifest}:{line_number}")
                self.boundaries[file_id] = (float(record["prompt_secs"]), int(record["prompt_frames"]))
        if not self.boundaries:
            raise ValueError(f"prompt boundary manifest is empty: {boundary_manifest}")

        dataset_file_ids = getattr(original_dataset, "fileid", None)
        if dataset_file_ids is not None:
            file_ids = [str(file_id) for file_id in dataset_file_ids]
            if len(file_ids) != len(set(file_ids)):
                raise ValueError("source audio manifest contains duplicate file IDs")
            missing = sorted(set(file_ids) - self.boundaries.keys())
            if missing:
                raise KeyError(f"no prompt boundary for {len(missing)} source files; examples: {missing[:5]}")

    def __len__(self) -> int:
        return len(tp.cast("tp.Sized", self.original_dataset))

    def __getitem__(self, index: int) -> tuple[Tensor, int, Path]:
        waveform, num_samples, file_path = self.original_dataset[index]
        if waveform is None:
            return tp.cast("tuple[Tensor, int, Path]", (waveform, num_samples, file_path))

        file_path = Path(file_path)
        file_id = file_path.stem
        if file_id not in self.boundaries:
            raise KeyError(f"no prompt boundary for {file_id!r}")
        prompt_seconds, prompt_frames = self.boundaries[file_id]
        prompt_samples = round(prompt_seconds * AUDIO_SAMPLE_RATE)
        if prompt_samples <= 0 or prompt_samples > int(num_samples):
            raise ValueError(
                f"prompt boundary for {file_id!r} requires {prompt_samples} samples; audio has {num_samples}"
            )

        waveform = normalize_waveform(waveform[:prompt_samples])
        required_samples = minimum_input_samples(prompt_frames, self.conv_layers)
        if waveform.numel() < required_samples:
            waveform = F.pad(waveform, (0, required_samples - waveform.numel()))
        actual_frames = int(conv_length(self.conv_layers, torch.tensor(waveform.numel())).item())
        if actual_frames != prompt_frames:
            raise ValueError(f"prompt crop for {file_id!r} produces {actual_frames} frames; expected {prompt_frames}")
        return waveform, waveform.numel(), file_path
