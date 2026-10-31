# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Encoder unit-extraction and k-means configuration."""

import typing as tp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ModelType = tp.Literal["spidr", "hubert", "wav2vec2", "wavlm"]


def _resolve_paths(paths: list[Path] | None) -> list[Path]:
    if paths is None:
        return []
    return [Path(path).resolve() for path in paths]


def check_folder_empty(folder_path: Path) -> None:
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    if len(list(folder_path.iterdir())) > 0:
        raise ValueError(f"The folder {folder_path} is not empty.")


def layer_num_str(layers: list[int]) -> str:
    # No layer ("") can be used when relevant for the feature extraction (e.g., text)
    return str(layers[0]) if layers else ""


@dataclass
class KMConfig:
    n_clusters: int
    seed: int = 0
    km_path: Path | None = None
    init: str = "k-means++"
    max_iter: int = 100
    batch_size: int = 10000
    tol: float = 0.0
    max_no_improvement: int = 100
    n_init: int = 20
    reassignment_ratio: float = 0.0
    delete_embs_folder: bool = True

    def __post_init__(self) -> None:
        """Resolve an optional pretrained k-means model path."""
        if self.km_path is not None:
            self.km_path = Path(self.km_path).expanduser().resolve()


@dataclass
class ManifestTarget:
    """Audio manifests for one downstream component."""

    manifests: list[Path]

    def __post_init__(self) -> None:
        self.manifests = _resolve_paths(self.manifests)


@dataclass
class GenerativeEvalManifest:
    """Evaluation audio and boundaries for independently encoded prompts."""

    audio_manifest: Path
    prompt_boundary_manifest: Path

    def __post_init__(self) -> None:
        self.audio_manifest = Path(self.audio_manifest).expanduser().resolve()
        self.prompt_boundary_manifest = Path(self.prompt_boundary_manifest).expanduser().resolve()


def _coerce_dataclass[T](value: T | dict[str, tp.Any], cls: type[T]) -> T:
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"{cls.__name__} configuration must be a mapping")
    return cls(**value)


@dataclass
class GenerativeEvalTarget:
    """Manifests requiring both full-utterance and prompt-only units."""

    manifests: list[GenerativeEvalManifest]

    def __post_init__(self) -> None:
        self.manifests = [_coerce_dataclass(value, GenerativeEvalManifest) for value in self.manifests]


@dataclass
class FeatureExtractionTargets:
    """Downstream consumers of framewise speech units."""

    duration_predictor: ManifestTarget | None = None
    unit_vocoder: ManifestTarget | None = None
    generative_eval: GenerativeEvalTarget | None = None

    def __post_init__(self) -> None:
        if self.duration_predictor is not None:
            self.duration_predictor = _coerce_dataclass(self.duration_predictor, ManifestTarget)
        if self.unit_vocoder is not None:
            self.unit_vocoder = _coerce_dataclass(self.unit_vocoder, ManifestTarget)
        if self.generative_eval is not None:
            self.generative_eval = _coerce_dataclass(self.generative_eval, GenerativeEvalTarget)


@dataclass
class FeatExtractConfig:
    checkpoint: Path | None
    model_type: ModelType
    output_folder: Path
    layers: list[int]
    manifests_train: list[Path] = field(default_factory=list)
    manifests_valid: list[Path] = field(default_factory=list)
    manifests_test: list[Path] = field(default_factory=list)
    targets: FeatureExtractionTargets | dict[str, tp.Any] | None = None
    output_format: Literal["auto", "txt", "jsonl"] = "auto"
    deduplicate: bool = True  # Deduplicate each unit sequence independently.
    cross_stream_deduplicate: bool = False  # Deduplicate aligned multi-stream codes jointly.
    chunk: bool = True  # Chunk unit sequences after optional deduplication.
    chunk_size: int = 2048  # Max length per chunk when chunk=True
    overwrite_folder: bool = False
    distributed_port: int = 58555
    batch_size: int = 8
    clustering: Literal["codes", "kmeans"] = "codes"
    manifests_train_kmeans: list[Path] | None = None
    vocab_size: int | list[int] = 256
    # Kmeans config
    kmeans: KMConfig | dict[str, tp.Any] | None = None
    kmeans_pooling_width: int = 1
    encoder_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.output_format not in {"auto", "txt", "jsonl"}:
            raise ValueError("output_format must be auto, txt, or jsonl")
        self.checkpoint = Path(self.checkpoint).resolve() if self.checkpoint is not None else None
        self.output_folder = Path(self.output_folder).resolve()
        self.manifests_train = _resolve_paths(self.manifests_train)
        self.manifests_valid = _resolve_paths(self.manifests_valid)
        self.manifests_test = _resolve_paths(self.manifests_test)
        self.manifests_train_kmeans = _resolve_paths(self.manifests_train_kmeans)
        if self.targets is not None:
            self.targets = _coerce_dataclass(self.targets, FeatureExtractionTargets)
            if self.manifests_train or self.manifests_valid or self.manifests_test:
                raise ValueError("targets cannot be combined with manifests_train, manifests_valid, or manifests_test")

        self.cfg_kmeans = self._create_kmeans_config()
        self._validate_layers()
        self._validate_deduplication()
        self._validate_kmeans()

        if self.kmeans_pooling_width < 1:
            raise ValueError("The 'kmeans_pooling_width' attribute must be a positive integer.")

    def _create_kmeans_config(self) -> KMConfig:
        n_clusters = (
            self.vocab_size if isinstance(self.vocab_size, int) else self.vocab_size[0] if self.vocab_size else 0
        )
        if self.kmeans is None:
            return KMConfig(n_clusters=n_clusters)
        if isinstance(self.kmeans, KMConfig):
            if self.kmeans.n_clusters != n_clusters:
                raise ValueError("kmeans.n_clusters must match vocab_size")
            return self.kmeans
        if "n_clusters" in self.kmeans:
            raise ValueError("Do not specify 'n_clusters' in kmeans config; it is derived from vocab_size.")
        return KMConfig(**self.kmeans, n_clusters=n_clusters)

    def _validate_layers(self) -> None:
        if not isinstance(self.layers, list):
            raise TypeError("The 'layers' attribute must be a list.")
        if len(self.layers) not in (0, 1):
            if self.clustering != "codes":
                raise ValueError(
                    "The 'layers' attribute must be a list of length 0 or 1 unless "
                    "clustering=='codes' (multi-stream codes extraction)."
                )
            if not isinstance(self.vocab_size, list) or len(self.vocab_size) != len(self.layers):
                raise ValueError(
                    "Multi-layer codes extraction requires 'vocab_size' to be a list with the "
                    f"same length as 'layers' (got vocab_size={self.vocab_size}, layers={self.layers})."
                )

    def _validate_deduplication(self) -> None:
        if self.is_multistream_codes:
            if self.deduplicate:
                raise ValueError(
                    "Multi-stream codes extraction requires 'deduplicate=False'. "
                    "Use 'cross_stream_deduplicate=True' for aligned dedup, or disable both for no dedup."
                )
        elif self.cross_stream_deduplicate:
            raise ValueError(
                "'cross_stream_deduplicate=True' is only valid for multi-stream codes extraction "
                "(clustering=='codes' with len(layers) > 1 and matching list 'vocab_size')."
            )

    def _validate_kmeans(self) -> None:
        km_path = self.cfg_kmeans.km_path
        if self.clustering != "kmeans":
            if km_path is not None:
                raise ValueError("kmeans.km_path is only valid when clustering='kmeans'")
            return
        if len(self.layers) != 1 or not isinstance(self.vocab_size, int):
            raise ValueError("k-means extraction requires one layer and an integer vocab_size")
        if km_path is not None and self.manifests_train_kmeans:
            raise ValueError("kmeans.km_path and manifests_train_kmeans are mutually exclusive")
        if km_path is None and not self.manifests_train_kmeans:
            raise ValueError("k-means fitting requires manifests_train_kmeans when kmeans.km_path is not set")

    @property
    def is_multistream_codes(self) -> bool:
        return self.clustering == "codes" and isinstance(self.vocab_size, list) and len(self.layers) > 1

    @property
    def effective_output_layers(self) -> list[int]:
        if self.is_multistream_codes:
            return [self.layers[0]]
        return self.layers
