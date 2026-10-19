"""Speech dataset and dataloader. Adapted from torchaudio."""

import abc
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import PathLike
from typing import override

import pandas as pd
import polars as pl
import torch
from polars.dataframe.frame import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from gslm2.config import AUDIO_SAMPLE_RATE, DataConfig
from gslm2.data.common.audio_io import decode_audio, normalize_waveform, reject_voxpopuli_ogg_paths
from gslm2.data.common.utils import bytes_from_archive, read_manifest

logger = logging.getLogger()

MERGE_CHUNK_THRESH = 100  # if remainder data lang chunk is greater than this (in seconds), a new chunk will be created

#: Columns that make a manifest row a tar byte-range rather than a path on disk.
ARCHIVE_COLUMNS = ("archive", "byte_offset", "byte_size")
_ARCHIVE_MANIFEST_COLUMNS = ("fileid", "path", "num_frames", *ARCHIVE_COLUMNS)


@dataclass(frozen=True)
class Sample:
    waveform: Tensor
    language: list[str | None] | None = None


@dataclass(frozen=True)
class SampleBatch:
    waveforms: Tensor
    languages: list[str | None] | None = None
    attention_mask: Tensor | None = None
    mask: Tensor | None = None

    def to(self, device: torch.device) -> "SampleBatch":
        return SampleBatch(
            waveforms=self.waveforms.to(device),
            languages=self.languages,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
        )


def get_number_of_data_chunks(manifest_file: str, duration_chunk_size: int | None) -> int:
    """
    Indicates number of data chunks in the manifest file across languages.

    manifest_file: str, path to the manifest file
    duration_chunk_size: int, desired language chunk size in seconds
    """
    manifest: DataFrame = read_manifest(manifest_file)
    if "language" not in manifest.columns:
        raise ValueError("Manifest must contain 'language' column for language chunking.")
    if not duration_chunk_size:
        return len(manifest["language"].unique())
    manifest = (
        manifest.group_by("language")
        .agg(pl.col("num_frames").sum())
        .with_columns((pl.col("num_frames") // (AUDIO_SAMPLE_RATE * duration_chunk_size)).alias("floor_chunks"))
        .with_columns(
            (pl.col("num_frames") % (AUDIO_SAMPLE_RATE * duration_chunk_size) > AUDIO_SAMPLE_RATE * MERGE_CHUNK_THRESH)
            .alias("remainder")
            .cast(int)
        )
    )
    return (manifest["floor_chunks"] + manifest["remainder"]).sum()


def divide_manifest_language_by_chunks(manifest: DataFrame, duration_chunk_size: int, seed: int) -> DataFrame:
    """
    Divide the manifest into chunks of a specified duration for each language.

    manifest: DataFrame with columns ["fileid", "path", "num_frames", "language"]
    duration_chunk_size is desired language chunk in seconds
    """
    chunked_manifest = []
    for lang in manifest["language"].unique(maintain_order=True):
        lang_manifest = manifest.filter(pl.col("language") == lang).sample(fraction=1.0, seed=seed, shuffle=True)
        frames_chunk = duration_chunk_size * AUDIO_SAMPLE_RATE
        assert lang_manifest["num_frames"].sum() >= frames_chunk, "Not enough frames for chunking language"
        lang_manifest = lang_manifest.with_columns(
            (pl.col("num_frames").cum_sum()).alias("cumsum_frames")
        ).with_columns((pl.lit(None, dtype=int)).alias("language_chunk"))
        frame_chunk_distances = range(0, lang_manifest["num_frames"].sum(), frames_chunk)
        for i, c in enumerate(frame_chunk_distances):
            lang_manifest = lang_manifest.with_columns(
                pl.when(pl.col("cumsum_frames") > c)
                .then(i)
                .otherwise(pl.col("language_chunk"))
                .alias("language_chunk")
            )
        max_chunk = lang_manifest["language_chunk"].max()
        if (
            lang_manifest.filter(pl.col("language_chunk") == max_chunk)["num_frames"].sum()
            < MERGE_CHUNK_THRESH * AUDIO_SAMPLE_RATE
        ):
            lang_manifest = lang_manifest.with_columns(
                pl.when(pl.col("language_chunk") == max_chunk)
                .then(max_chunk - 1)
                .otherwise(pl.col("language_chunk"))
                .alias("language_chunk")
            )
        lang_manifest = lang_manifest.with_columns(
            pl.format("{}_chunk{}", pl.col("language"), pl.col("language_chunk")).alias("language")
        )
        chunked_manifest.append(lang_manifest)
    chunked_manifest = pl.concat(chunked_manifest)
    chunked_manifest = chunked_manifest.drop(["cumsum_frames", "language_chunk"])
    logger.info("Split data into %s chunks", chunked_manifest["language"].n_unique())
    return chunked_manifest


class SpeechDataset(Dataset, abc.ABC):
    """Dataset to load chunks of audio files."""

    def __init__(self, data_cfg: DataConfig) -> None:
        super().__init__()
        self._lance_uri: str | None = None
        self._lance_storage_options: dict | None = None
        self._prefetch_threads: int = 0
        self._pool: ThreadPoolExecutor | None = None
        self.lance_indices: list[int] | None = None
        self.normalize = data_cfg.normalize
        self.manifest = self._load_manifest(data_cfg)

        if data_cfg.vad is not None:
            raise NotImplementedError(
                "DataConfig.vad (VAD-synthesis mode) is not supported in gslm2: "
                "pre-segment the audio and point DataConfig.manifest at the segment manifest."
            )
        self._init_from_manifest(data_cfg)

    def _load_manifest(self, data_cfg: DataConfig) -> pl.DataFrame:
        if data_cfg.backend == "lance":
            manifest, self._lance_uri, self._lance_storage_options = _lance_audio_manifest(data_cfg.manifest)
        elif data_cfg.manifest:
            manifest = read_manifest(data_cfg.manifest)
        else:
            raise ValueError("DataConfig.manifest is required")
        # Archive rows decode a whole tar byte range, so the unreliable seek timeline never applies.
        if "path" in manifest.columns and not all(col in manifest.columns for col in ARCHIVE_COLUMNS):
            reject_voxpopuli_ogg_paths(manifest["path"], data_cfg.manifest)
        if data_cfg.lang_chunk_size is not None:
            manifest = divide_manifest_language_by_chunks(
                manifest, data_cfg.lang_chunk_size, seed=data_cfg.random_seed
            )
        return manifest

    def _init_from_manifest(self, data_cfg: DataConfig) -> None:
        if data_cfg.sort_by_length:
            self.manifest = self.manifest.sort("num_frames", descending=True)
        # Lance blobs live on S3EOZ and are round-trip-latency-bound; overlap a bucketed batch's
        # reads across a per-worker thread pool (see __getitems__). Off for local/archive backends.
        self._prefetch_threads = data_cfg.lance_prefetch_threads if data_cfg.backend == "lance" else 0
        self._pool = None
        self.fileid = list(self.manifest["fileid"])
        self.filenames = list(self.manifest["path"])
        self.lengths = list(self.manifest["num_frames"])
        self.lang = list(self.manifest["language"]) if "language" in self.manifest else None
        # Lance rows are addressed by their original row index, which survives any sort above.
        self.lance_indices = list(self.manifest["lance_index"]) if data_cfg.backend == "lance" else None

    def __len__(self) -> int:
        return len(self.fileid)

    @abc.abstractmethod
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        pass

    def __getitems__(self, indices: list[int]) -> list["Sample"]:
        """Batch fetch used by torch's DataLoader when a batch_sampler is set. With
        ``lance_prefetch_threads > 1`` the batch's reads run concurrently on a per-worker
        (lazily created) thread pool — the S3 I/O + torchcodec decode release the GIL, so the
        round-trips overlap. Order is preserved; falls back to serial when disabled."""
        if self._prefetch_threads <= 1 or len(indices) <= 1:
            return [self[i] for i in indices]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._prefetch_threads)
        return list(self._pool.map(self.__getitem__, indices))

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)  # getattr: __init__ may not have completed
        if pool is not None:
            pool.shutdown(wait=False)

    def __getitem__(self, index: int) -> Sample:
        waveform, sr = self._load_audio(index)
        if sr != AUDIO_SAMPLE_RATE or waveform.shape[0] != 1:
            raise ValueError(index)
        if self.normalize:
            waveform = normalize_waveform(waveform)
        language = self.lang[index] if self.lang else None
        return Sample(
            waveform=waveform.squeeze(),
            language=language,
        )


class SpeechDatasetFromArchive(SpeechDataset):
    def __init__(self, data_cfg: DataConfig) -> None:
        super().__init__(data_cfg)
        manifest = self.manifest.to_dict()
        self.archives = list(manifest["archive"])
        self.byte_offsets = list(manifest["byte_offset"])
        self.byte_sizes = list(manifest["byte_size"])

    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        archive, offset, size = self.archives[index], self.byte_offsets[index], self.byte_sizes[index]
        data = bytes_from_archive(archive, offset, size)
        return decode_audio(data)


class SpeechDatasetFromFiles(SpeechDataset):
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        return decode_audio(self.filenames[index])


class SpeechDatasetFromFilesInference(SpeechDatasetFromFiles):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Overwrite as the lengths of waveforms are needed to create the masks for batching"""
        waveform, sr = self._load_audio(index)
        if sr != AUDIO_SAMPLE_RATE or waveform.shape[0] != 1:
            raise ValueError(index)
        if self.normalize:
            waveform = normalize_waveform(waveform)
        return waveform.squeeze(), waveform.shape[-1], self.filenames[index]


class SpeechDatasetFromFilesInferenceSafe(SpeechDatasetFromFilesInference):
    @override
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        return decode_audio(self.filenames[index])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        try:
            return super().__getitem__(index)
        except Exception as e:
            logger.exception("Error loading %s: %s", self.filenames[index], e)  # noqa: TRY401
            logger.warning("Returning 'None' for this sample.")
            return None, None, self.filenames[index]  # type: ignore[return-value]


class SpeechDatasetFromArchiveInference(SpeechDatasetFromArchive):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        waveform, sr = self._load_audio(index)
        if sr != AUDIO_SAMPLE_RATE or waveform.shape[0] != 1:
            raise ValueError(index)
        if self.normalize:
            waveform = normalize_waveform(waveform)
        return waveform.squeeze(), waveform.shape[-1], self.filenames[index]


class SpeechDatasetFromArchiveInferenceSafe(SpeechDatasetFromArchiveInference):
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        try:
            return super().__getitem__(index)
        except Exception as e:
            logger.exception("Error loading %s: %s", self.filenames[index], e)  # noqa: TRY401
            logger.warning("Returning 'None' for this sample.")
            return None, None, self.filenames[index]  # type: ignore[return-value]


def _is_archive_manifest(manifest_path: PathLike) -> bool:
    manifest = pd.read_csv(manifest_path, nrows=1)
    return all(col in manifest.columns for col in _ARCHIVE_MANIFEST_COLUMNS)


def speech_dataset_inference_safe(data_cfg: DataConfig) -> SpeechDataset:
    if _is_archive_manifest(data_cfg.manifest):
        return SpeechDatasetFromArchiveInferenceSafe(data_cfg)
    return SpeechDatasetFromFilesInferenceSafe(data_cfg)


def _resolve_lance_uri(spec: str) -> str:
    """Resolve a Lance audio manifest spec (raw URI, dataset-card name, or ``dataset:enrich-key``)."""
    from gslm2.data.registry import resolve_manifest_uri

    return resolve_manifest_uri(spec)


def _lance_audio_manifest(spec: str) -> tuple[pl.DataFrame, str, dict | None]:
    """Build a (fileid, path, num_frames, lance_index) manifest from a Lance audio dataset."""
    import lance

    from gslm2.data.lance_store import storage_options
    from gslm2.data.s3_creds import ensure_s3_credentials

    uri = _resolve_lance_uri(spec)
    ensure_s3_credentials(uri)
    so = storage_options(uri)
    columns = lance.dataset(uri, storage_options=so).to_table(columns=["recording_id", "num_frames"]).to_pydict()
    num_rows = len(columns["recording_id"])
    manifest = pl.DataFrame(
        {
            "fileid": columns["recording_id"],
            "path": columns["recording_id"],
            "num_frames": columns["num_frames"],
            "lance_index": list(range(num_rows)),
        }
    )
    return manifest, uri, so


class SpeechDatasetFromLance(SpeechDataset):
    """Reads audio from a Lance dataset's ``audio_mono`` blob instead of loose files."""

    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        from gslm2.data.common.audio_io import decode_audio
        from gslm2.data.lance_grain import _DS_CACHE, _open_dataset
        from gslm2.data.lance_store import open_audio_blob
        from gslm2.data.s3_creds import ensure_s3_credentials, start_credential_refresh

        ensure_s3_credentials(self._lance_uri)  # idempotent; renews env in spawn workers
        start_credential_refresh(self._lance_uri, on_refresh=_DS_CACHE.clear)
        dataset = _open_dataset(self._lance_uri, self._lance_storage_options)
        # Decode the FLAC ``audio_mono`` blob with torchcodec via the shared ``decode_audio`` (the
        # same decode the file/archive speech datasets use), returning ``(waveform [C, T] float32,
        # sample_rate)``.
        with open_audio_blob(dataset, "audio_mono", self.lance_indices[index]) as blob:
            return decode_audio(blob.read())


def speech_dataset(data_cfg: DataConfig) -> SpeechDataset:
    if data_cfg.backend == "lance":
        return SpeechDatasetFromLance(data_cfg)
    # VAD-synthesis, no manifest to probe, audio comes from disk via VADLoader
    if not data_cfg.manifest:
        return SpeechDatasetFromFiles(data_cfg)
    if _is_archive_manifest(data_cfg.manifest):
        return SpeechDatasetFromArchive(data_cfg)
    return SpeechDatasetFromFiles(data_cfg)
