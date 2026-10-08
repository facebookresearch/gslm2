"""Data loading and augmentation configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LMTask = Literal["causal"]


@dataclass(frozen=True)
class AugmentConfig:
    """Data augmentation configuration."""

    name: Literal["pitch_shift", "noise_mix"]
    prob: float = 0.0
    augmentation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig:
    """Dataloading and data sampling configuration."""

    manifest: str | None = None
    #: Audio backend: "file" (loose/archive from the CSV manifest) or "lance" (manifest is a
    #: Lance dataset-card name or s3:// URI; audio read from the audio_mono blob).
    backend: str = "file"
    enable_padding: bool = False
    rand_crop: bool = True
    normalize: bool = True
    num_buckets: int = 1000
    min_sample_size: int = 2_000
    max_sample_size: int = 320_000
    max_batch_length: int = 3_800_000
    drop_last: bool = False
    num_workers: int = 10
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    random_seed: int = 0
    bucket_method: Literal["uniform", "percentile"] = "uniform"
    alignments_path: Path | None = None
    by_lang: bool = False
    lang_chunk_size: int | None = None
    sort_by_length: bool = False
    #: Lance backend only: concurrent within-batch blob reads per worker (via ``__getitems__``).
    #: Small S3EOZ audio blobs are round-trip-latency-bound, so overlapping a bucketed batch's
    #: reads across a thread pool (GIL released during S3 I/O + soundfile decode) hides the
    #: latency without adding worker processes. 0/1 = serial (default; identical to plain reads).
    lance_prefetch_threads: int = 0
    enable_packing: bool = False
    sampler_type: Literal["bucketize", "packing"] = "bucketize"
    # Optional VAD-synthesis config (internal segmentation pipeline); None reads the manifest.
    vad: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataConfig":
        """Build from a plain dict (YAML / ``asdict`` round-trip)."""
        return cls(**dict(d))


@dataclass(frozen=True)
class SpeechUnitConfig:
    """Frozen-encoder settings for on-the-fly audio->unit SLM training.

    The encoder is loaded from ``checkpoint`` and run in eval/no-grad to produce
    discrete units (see ``gslm2.data.speech.unit_extraction``). ``unit_offset``
    reserves ids below it (PAD=0, EOS=1), so the LM vocab is
    ``codebook_size + unit_offset``. ``model_type`` selects the encoder in the model
    registry (``spidr`` / ``dinosr``).
    """

    checkpoint: str
    model_type: str = "spidr"
    extract_layer: int = 6
    codebook_size: int = 256
    unit_offset: int = 2
    deduplicate: bool = True
    max_token_seq_len: int = 2048
    encoder_micro_batch: int = 10
    dtype: str = "float32"
    # If set, per-file units are cached to this shared directory (keyed by dataset index) on
    # first encode and read back thereafter — every rank fills a disjoint shard in epoch 1, so
    # epoch 2+ is all cache hits regardless of how the sampler reshards. Requires rand_crop=False
    # (deterministic crop) for the cached units to stay valid across epochs.
    unit_cache_dir: str | None = None


@dataclass(frozen=True)
class OnTheFlyUnitSource:
    """On-the-fly SLM source: an audio manifest plus a frozen encoder that extracts units in the
    training step (see :class:`SpeechUnitConfig`). Single-stream today — the extractor picks one
    codebook layer and deduplicates (``gslm2.data.speech.unit_extraction``).
    """

    audio: DataConfig
    feature_extraction: SpeechUnitConfig


@dataclass(frozen=True)
class PreExtractedUnitSource:
    """Pre-extracted SLM source: ``.txt`` manifests of raw codebook ids, one line per utterance with
    the ``num_streams`` aligned streams concatenated as equal-length space-separated blocks (stream 0's
    tokens, then stream 1's, ...), offset by ``unit_offset`` at load (reserving PAD=``pad_idx`` and other
    reserved ids).
    """

    manifests: list[str] = field(default_factory=list)
    manifests_weights: list[float] = field(default_factory=lambda: [1.0])
    unit_offset: int = 2
    pad_idx: int = 1
    max_seq_len: int = 4096
    min_seq_len: int = 2
    # Sequence packing: concatenate several documents per row (block-diagonal attention) to remove
    # padding waste. ``batch_size`` then counts packed rows, so raise it when enabling packing.
    pack: bool = False
    batch_size: int = 32
    drop_last: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    random_seed: int = 0

    def __post_init__(self) -> None:
        # Each manifest needs its own sampling weight; the loader zips the two with strict=True
        # (build_multistream_unit_dataloader), so a mismatch would otherwise crash late and cryptically.
        if len(self.manifests_weights) != len(self.manifests):
            raise ValueError(
                f"manifests_weights length ({len(self.manifests_weights)}) "
                f"must match manifests length ({len(self.manifests)})"
            )


@dataclass(frozen=True)
class SLMUnitDataConfig:
    """Speech-unit LM data over ``num_streams`` aligned streams from one ``source``.

    ``source`` is either on-the-fly audio->unit extraction (:class:`OnTheFlyUnitSource`, single-stream)
    or pre-extracted unit manifests (:class:`PreExtractedUnitSource`, single- or multi-stream). The
    stream count must match the model's (``len(vocab_sizes)`` for ``MultiStreamCausalLMConfig``).
    """

    source: OnTheFlyUnitSource | PreExtractedUnitSource
    num_streams: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.source, OnTheFlyUnitSource) and self.num_streams != 1:
            raise NotImplementedError(
                "on-the-fly audio extraction currently yields a single stream, so num_streams > 1 "
                "requires pre-extracted units (PreExtractedUnitSource). Multi-stream on-the-fly "
                "extraction (one stream per SpidR codebook layer, plus cross-stream deduplication: a "
                "single shared keep-mask drops a frame only when every stream is static there, so the "
                "streams stay aligned and equal-length) is not yet wired."
            )


@dataclass(frozen=True)
class TextDataConfig:
    """Configuration for text dataloading from ``.txt`` / ``.json`` manifests."""

    manifests: list[str] = field(default_factory=list)
    manifests_weights: list[float] = field(default_factory=lambda: [1.0])
    tokenizer: str = "gpt2"
    task: LMTask = "causal"
    max_length: int = 512
    batch_size: int = 32
    text_key: str = "text"
    add_special_tokens: bool = True
    drop_last: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    packing: bool = False
    random_seed: int = 0
