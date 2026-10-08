# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import math
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import joblib
import torch
import torch.distributed as distr
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gslm2.config import DataConfig, DinoSRConfig
from gslm2.data.common.masking.speech import build_attention_mask
from gslm2.data.speech.collators import conv_length
from gslm2.data.speech.dataset import speech_dataset_inference_safe
from gslm2.eval.slm.config import FeatExtractConfig
from gslm2.eval.slm.create_pairs import create_paired_file
from gslm2.eval.slm.distributed import DistributedContext
from gslm2.eval.slm.learn_kmeans import learn_kmeans
from gslm2.eval.slm.utils import timing
from gslm2.models import build_model

logger = logging.getLogger(__name__)

OutputType = tp.Literal["codes", "embs", "embs_clustered_kmeans"]

CHUNK_SIZE = 2048
KMEANS_CENTERS_NDIM = 2


@dataclass(frozen=True)
class RowFormat:
    chunk: bool
    chunk_size: int
    output_format: str


def get_extractor_conv_layer_config() -> list[tuple[int, int, int]]:
    # TODO: config should be read from the config file in same location as the SSL ckpt  # noqa: FIX002, TD002
    # Same default values for HuBert, Wav2vec2, DinoSR, SpidR
    # https://huggingface.co/spaces/gradio/HuBERT/blob/main/fairseq/models/wav2vec/wav2vec2.py#L94
    # https://huggingface.co/spaces/gradio/HuBERT/blob/main/fairseq/models/hubert/hubert.py
    # https://github.com/microsoft/unilm/blob/c45389eda88e14c57de2c07472e3f49383a6dab0/wavlm/WavLM.py#L173C40-L173C92

    return DinoSRConfig().extractor_conv_layer_config


class ApplyKmeans:
    def __init__(self, km_path: str | Path, expected_n_clusters: int) -> None:
        self.km_model = joblib.load(km_path)
        centers = self.km_model.cluster_centers_
        if centers.ndim != KMEANS_CENTERS_NDIM:
            raise ValueError(f"expected rank-2 k-means centers, got shape {centers.shape}")
        if centers.shape[0] != expected_n_clusters:
            raise ValueError(f"expected {expected_n_clusters} k-means clusters, got {centers.shape[0]} from {km_path}")
        self.C_np = centers.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.C.size(0):
            raise ValueError(
                f"k-means feature dimension mismatch: extractor produced {x.size(-1)}, "
                f"but model expects {self.C.size(0)}"
            )
        dist = (
            x.pow(2).sum(-1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        )  # batch_size x seq x n_clusters
        return dist.argmin(dim=-1)


class StridedDataset(Dataset):
    def __init__(self, original_dataset: Dataset, rank: int, stride: int) -> None:
        self.original_dataset = original_dataset
        self.stride = stride
        self.rank = rank

    def __len__(self) -> int:
        n_samples = len(self.original_dataset) // self.stride
        remainder = len(self.original_dataset) % self.stride
        return n_samples + (1 if remainder > self.rank else 0)

    def __getitem__(self, idx: int) -> list[tp.Any]:
        i = idx * self.stride + self.rank
        return self.original_dataset[i]


def stringify_units(units: list[int]) -> str:
    return " ".join(map(str, units))


def split_into_chunks(lst: list[tp.Any], chunk_size: int = CHUNK_SIZE) -> list[list[tp.Any]]:
    """
    Split a list into chunks of a given size.
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_row(  # noqa: PLR0913
    row: torch.Tensor,
    output_length: int,
    file_path: Path,
    output_format: str,
    *,
    deduplicate: bool,
    chunk: bool,
    chunk_size: int = CHUNK_SIZE,
) -> str | dict[str, str] | list[str]:
    units_row = row[: int(output_length)]
    if deduplicate:
        units_row = torch.unique_consecutive(units_row)

    if output_format == "jsonl":
        audio_repr = stringify_units(units_row.tolist())
        out = {"file_name": str(file_path), "audio_repr": audio_repr}
    elif output_format == "txt":
        out = split_into_chunks(units_row.tolist(), chunk_size=chunk_size) if chunk else [units_row.tolist()]
        out = [stringify_units(e) for e in out]
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")
    return out


@torch.no_grad()
def aligned_dedup_keep_mask(stream_rows: list[torch.Tensor]) -> torch.Tensor:
    """Keep frames where any aligned stream changes value."""
    if not stream_rows:
        raise ValueError("aligned_dedup_keep_mask requires at least one stream row.")
    length = stream_rows[0].numel()
    device = stream_rows[0].device
    if length == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    stacked = torch.stack([row.view(-1) for row in stream_rows], dim=0)
    keep = torch.ones(length, dtype=torch.bool, device=device)
    if length > 1:
        keep[1:] = (stacked[:, 1:] != stacked[:, :-1]).any(dim=0)
    return keep


def save_embeddings_batched(  # noqa: PLR0913
    embeddings: list[torch.tensor],
    attention_mask: torch.Tensor | None,
    layers: list[int],
    output_dir: Path,
    split: str,
    file_name: str,
) -> None:
    for layer in layers:
        output_file = (output_dir / f"{layer}/{split}/{file_name}").with_suffix(".pt")
        output_file.parent.mkdir(exist_ok=True, parents=True)

        embs_flat = embeddings[layer - 1].view(-1, embeddings[layer - 1].shape[-1])

        if attention_mask is None:
            torch.save(embs_flat.float().cpu(), output_file)
        else:
            mask_flat = attention_mask.squeeze(1)[:, 0, :].flatten()
            torch.save(embs_flat[mask_flat].float().cpu(), output_file)


def save_units_batched(  # noqa: PLR0913
    outputs: list[torch.tensor],
    output_type: OutputType,
    conv_lengths: torch.tensor,
    file_paths: list[Path],
    layers: list[int],
    shard_files: dict,
    kmeans_model: ApplyKmeans | None = None,
    *,
    deduplicate: bool = True,
    chunk: bool = True,
    chunk_size: int = CHUNK_SIZE,
    output_format: str = "txt",
) -> None:
    for layer in layers:
        if output_type == "codes":
            units = outputs[layer - 1].argmax(dim=-1)
        elif output_type == "embs_clustered_kmeans":
            units = kmeans_model(outputs[layer - 1])

        if units.dim() == 1:
            units = units.unsqueeze(0)
        rows = [
            process_row(
                row,
                length,
                file_path,
                deduplicate=deduplicate,
                output_format=output_format,
                chunk=chunk,
                chunk_size=chunk_size,
            )
            for row, length, file_path in zip(units, conv_lengths, file_paths, strict=True)
        ]
        if output_format == "jsonl":
            for row in rows:
                print(json.dumps(row), file=shard_files[layer])
        else:
            rows = [item for sublist in rows for item in sublist]
            shard_files[layer].writelines(f"{row}\n" for row in rows)


def rows_from_dedup_rows(
    dedup_rows: list[torch.Tensor],
    file_paths: list[Path],
    row_format: RowFormat,
) -> list[str | dict[str, str] | list[str]]:
    return [
        process_row(
            row,
            row.numel(),
            file_path,
            deduplicate=False,
            output_format=row_format.output_format,
            chunk=row_format.chunk,
            chunk_size=row_format.chunk_size,
        )
        for row, file_path in zip(dedup_rows, file_paths, strict=True)
    ]


def rows_from_units(
    units: torch.Tensor,
    conv_lengths: torch.tensor,
    file_paths: list[Path],
    *,
    deduplicate: bool,
    row_format: RowFormat,
) -> list[str | dict[str, str] | list[str]]:
    return [
        process_row(
            row,
            length,
            file_path,
            deduplicate=deduplicate,
            output_format=row_format.output_format,
            chunk=row_format.chunk,
            chunk_size=row_format.chunk_size,
        )
        for row, length, file_path in zip(units, conv_lengths, file_paths, strict=True)
    ]


def append_aligned_rows(
    rows: list[str | dict[str, str] | list[str]],
    next_rows: list[str | dict[str, str] | list[str]],
    *,
    output_format: str,
) -> None:
    for row, next_row in zip(rows, next_rows, strict=True):
        assert len(row) == len(next_row)
        if output_format == "jsonl":
            assert row["file_name"] == next_row["file_name"]
            row["audio_repr"] += "|" + next_row["audio_repr"]
        else:
            for chunk_idx in range(len(row)):
                row[chunk_idx] += " " + next_row[chunk_idx]


def deduplicate_aligned_units(
    units_by_layer: list[torch.Tensor], conv_lengths: torch.Tensor
) -> list[list[torch.Tensor]]:
    rows_by_layer: list[list[torch.Tensor]] = [[] for _ in units_by_layer]
    for batch_idx in range(units_by_layer[0].shape[0]):
        length = int(conv_lengths[batch_idx])
        layer_rows = [units[batch_idx, :length] for units in units_by_layer]
        keep = aligned_dedup_keep_mask(layer_rows)
        for rows, layer_row in zip(rows_by_layer, layer_rows, strict=True):
            rows.append(layer_row[keep])
    return rows_by_layer


def merge_aligned_rows(
    rows_by_layer: list[list[str | dict[str, str] | list[str]]], output_format: str
) -> list[str | dict[str, str] | list[str]]:
    rows = rows_by_layer[0]
    for next_rows in rows_by_layer[1:]:
        append_aligned_rows(rows, next_rows, output_format=output_format)
    return rows


def write_rows(rows: list[str | dict[str, str] | list[str]], shard_file: tp.TextIO, output_format: str) -> None:
    if output_format == "jsonl":
        for row in rows:
            print(json.dumps(row), file=shard_file)
        return

    flattened_rows = [item for row in rows for item in row]
    shard_file.writelines(f"{row}\n" for row in flattened_rows)


def save_multistream_codes_units_batched(  # noqa: PLR0913
    outputs: list[torch.tensor],
    conv_lengths: torch.tensor,
    file_paths: list[Path],
    layers: list[int],
    shard_files: dict,
    *,
    deduplicate: bool = True,
    cross_stream_deduplicate: bool = False,
    chunk: bool = True,
    chunk_size: int = CHUNK_SIZE,
    output_format: str = "txt",
) -> None:
    """Write aligned multi-layer code streams to the first layer's shard.

    Dedup modes (all keep the streams aligned and equal-length via a single shared keep-mask):
    - none: full-resolution frames.
    - ``cross_stream_deduplicate``: drop frames where *every* stream is static.
    """
    if deduplicate:
        raise ValueError("Multi-stream codes require deduplicate=False; use cross_stream_deduplicate instead.")

    output_layer = layers[0]
    units_by_layer = [outputs[layer - 1].argmax(dim=-1) for layer in layers]
    units_by_layer = [units.unsqueeze(0) if units.dim() == 1 else units for units in units_by_layer]

    row_format = RowFormat(chunk=chunk, chunk_size=chunk_size, output_format=output_format)

    if cross_stream_deduplicate:
        dedup_rows_by_layer = deduplicate_aligned_units(units_by_layer, conv_lengths)
        rows_by_layer = [rows_from_dedup_rows(rows, file_paths, row_format) for rows in dedup_rows_by_layer]
    else:
        rows_by_layer = [
            rows_from_units(units, conv_lengths, file_paths, deduplicate=False, row_format=row_format)
            for units in units_by_layer
        ]

    rows = merge_aligned_rows(rows_by_layer, output_format)
    write_rows(rows, shard_files[output_layer], output_format)


def worker_shard_path(  # noqa: PLR0913
    output_folder: Path,
    layer: int,
    split: str,
    dataset_name: str,
    rank: int,
    ext: str = "jsonl",
) -> Path:
    fname = output_folder / str(layer) / split / f"{dataset_name}_partial_{rank!s}.{ext}"
    fname.parent.mkdir(parents=True, exist_ok=True)
    return fname


def pad_collate(batch: list[tuple[object, ...]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_filtered = []
    for item in batch:
        if all(elem is not None for elem in item):
            batch_filtered.append(item)
        else:
            logger.warning("Collate: skipping erroneous None input.")
    if not batch_filtered:
        return None, None, None  # type: ignore  # noqa: PGH003
    batch = batch_filtered
    x, lengths, file_paths = zip(*batch)  # noqa: B905
    x = pad_sequence(x, batch_first=True, padding_value=0)
    file_paths = [Path(p) for p in file_paths]
    return x, torch.tensor(lengths), file_paths


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move2gpu(x: torch.Tensor) -> torch.Tensor:
    # no pin_memory here since apparently x can be on GPU already
    return x.to(_DEVICE)


def _device_scope(context: DistributedContext, n_gpus: int):  # noqa: ANN202
    import contextlib

    if torch.cuda.is_available() and n_gpus > 0:
        return torch.cuda.device(context.local_rank % n_gpus)
    return contextlib.nullcontext()


def apply_pooling(
    embs: list[torch.tensor], attention_mask: torch.Tensor | None, conv_lengths: torch.tensor, cfg: FeatExtractConfig
) -> tuple[list[torch.tensor], torch.Tensor | None, torch.tensor]:
    layer = cfg.layers[0]  # Assume only one layer
    k = cfg.kmeans_pooling_width
    batch_size, seq_len, feat_dim = embs[layer - 1].shape

    seq_len_out = math.ceil(seq_len / k)
    pad_len = (seq_len_out * k) - seq_len
    if pad_len > 0:
        last_emb = embs[layer - 1][:, -1:, :].expand(batch_size, pad_len, feat_dim)
        embs[layer - 1] = torch.cat([embs[layer - 1], last_emb], dim=1)

        # attention_mask is None when the batch has no padding (nothing to extend).
        if attention_mask is not None:
            pad_rows = attention_mask[:, :, -1:, :].expand(batch_size, 1, pad_len, attention_mask.size(-1))
            attention_mask = torch.cat([attention_mask, pad_rows], dim=2)
            pad_cols = attention_mask[:, :, :, -1:].expand(batch_size, 1, attention_mask.size(-2), pad_len)
            attention_mask = torch.cat([attention_mask, pad_cols], dim=3)

    x = embs[layer - 1].reshape(batch_size, seq_len_out, k, feat_dim)

    embs[layer - 1] = x.mean(dim=2)  # Mean pooling

    if attention_mask is not None:
        att_mask = attention_mask.reshape(batch_size, 1, seq_len_out, k, seq_len_out, k)
        attention_mask = att_mask.any(dim=3).any(dim=-1)
    conv_lengths = (conv_lengths / k).ceil().long()

    return embs, attention_mask, conv_lengths


@torch.no_grad()
def transcribe_manifest(  # noqa: C901, PLR0912, PLR0913, PLR0915
    cfg: FeatExtractConfig,
    batch_size: int,
    manifest: Path,
    manifest_split: str,
    rank: int,
    world_size: int,
    output_type: OutputType,
    kmeans_path: Path | None = None,
) -> None:
    # Model
    if cfg.model_type == "hubert":
        from torchaudio.pipelines import HUBERT_BASE

        model = HUBERT_BASE.get_model().to(_DEVICE).eval()
    elif cfg.model_type == "wav2vec2":
        from torchaudio.pipelines import WAV2VEC2_BASE

        model = WAV2VEC2_BASE.get_model().to(_DEVICE).eval()
    elif cfg.model_type == "wavlm":
        from torchaudio.pipelines import WAVLM_BASE

        model = WAVLM_BASE.get_model().to(_DEVICE).eval()
    else:
        model = build_model(checkpoint=cfg.checkpoint, model_type=cfg.model_type).to(_DEVICE).eval()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    encoder_dtype = dtype_map.get(cfg.encoder_dtype, torch.float32)
    model = model.to(dtype=encoder_dtype)

    extractor_conv_layer_config = get_extractor_conv_layer_config()
    causal_fe = getattr(getattr(model, "cfg", None), "causal_fe", False)

    if output_type == "embs_clustered_kmeans":
        if kmeans_path is None or not isinstance(cfg.vocab_size, int):
            raise ValueError("k-means extraction requires a model path and an integer vocab_size")
        kmeans_model = ApplyKmeans(kmeans_path, cfg.vocab_size)
    else:
        kmeans_model = None

    # Data
    cfg_dataset = DataConfig(manifest=manifest, normalize=True, sort_by_length=True)
    dataset = speech_dataset_inference_safe(cfg_dataset)
    dataset_name = manifest.stem

    strided_dataset = StridedDataset(
        original_dataset=dataset,
        rank=rank,
        stride=world_size,
    )
    logger.info("Rank %d: len(strided_dataset)= %d", rank, len(strided_dataset))

    dataloader = DataLoader(
        strided_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=pad_collate,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=False,
    )

    # For test data and unit_vocoder data, we keep file_path info to ensure correct downstream processing
    output_format = "jsonl" if manifest_split in ["test", "unit_vocoder"] else "txt"

    if output_type in ["codes", "embs_clustered_kmeans"]:
        shard_files = {
            layer: open(  # noqa: PTH123, SIM115
                worker_shard_path(cfg.output_folder, layer, manifest_split, dataset_name, rank, output_format), "w"
            )
            for layer in cfg.effective_output_layers
        }

    logger.info("Starting inference on rank: %d", rank)

    torch.empty(1).to(_DEVICE)
    total_files = 0
    total_batches = 0
    has_file_errors = False
    for sample_files in tqdm(dataloader, disable=rank != 0, desc=dataset_name):
        if any(elem is None for elem in sample_files):
            logger.warning("Skipping empty batch.")
            continue
        waveform_batch, lengths_batch, file_paths_batch = sample_files
        assert waveform_batch.shape[0] <= batch_size
        assert max(lengths_batch) == waveform_batch.shape[1]
        # Try the full batch first, then retry as single-file micro-batches on OOM.
        bz = batch_size
        while bz >= 1:
            assert bz in (batch_size, 1), f"Batch size should be either {batch_size} or 1, got {bz}"
            torch.cuda.empty_cache()
            oom_retry = False
            for batch_start in range(0, waveform_batch.shape[0], bz):
                try:
                    lengths = move2gpu(lengths_batch[batch_start : batch_start + bz])
                    waveform = move2gpu(waveform_batch[batch_start : batch_start + bz, : max(lengths)])
                    waveform = waveform.to(dtype=encoder_dtype)
                    file_paths = file_paths_batch[batch_start : batch_start + bz]
                    conv_lengths = conv_length(extractor_conv_layer_config, lengths, causal=causal_fe)
                    attention_mask = build_attention_mask(conv_lengths)

                    if output_type == "codes":
                        onehots = model.get_codebooks(
                            waveform,
                            attention_mask=attention_mask,
                            layers=cfg.layers,
                            onehot=True,
                        )
                        # this works because either the save worked on the entire batch
                        # or it will be applied on individual files (but not both)

                        if cfg.is_multistream_codes:
                            save_multistream_codes_units_batched(
                                outputs=onehots,
                                conv_lengths=conv_lengths,
                                file_paths=file_paths,
                                layers=cfg.layers,
                                shard_files=shard_files,
                                deduplicate=cfg.deduplicate,
                                cross_stream_deduplicate=cfg.cross_stream_deduplicate,
                                chunk=cfg.chunk,
                                chunk_size=cfg.chunk_size,
                                output_format=output_format,
                            )
                        else:
                            save_units_batched(
                                outputs=onehots,
                                output_type=output_type,
                                conv_lengths=conv_lengths,
                                file_paths=file_paths,
                                layers=cfg.layers,
                                shard_files=shard_files,
                                deduplicate=cfg.deduplicate,
                                chunk=cfg.chunk,
                                chunk_size=cfg.chunk_size,
                                output_format=output_format,
                            )
                    else:
                        if cfg.model_type in ["hubert", "wav2vec2", "wavlm"]:
                            embs, _ = model.extract_features(
                                waveforms=waveform,
                                lengths=lengths,
                            )
                        else:
                            embs = model.get_intermediate_outputs(
                                waveform, attention_mask=attention_mask, layers=cfg.layers
                            )

                        if cfg.kmeans_pooling_width > 1:
                            embs, attention_mask, conv_lengths = apply_pooling(embs, attention_mask, conv_lengths, cfg)

                        if output_type == "embs":
                            save_embeddings_batched(
                                embeddings=embs,
                                attention_mask=attention_mask,
                                layers=cfg.layers,
                                output_dir=cfg.output_folder,
                                split=manifest_split,
                                file_name=Path(file_paths[0]).stem,
                            )
                        elif output_type == "embs_clustered_kmeans":
                            save_units_batched(
                                outputs=embs,
                                output_type=output_type,
                                conv_lengths=conv_lengths,
                                file_paths=file_paths,
                                kmeans_model=kmeans_model,
                                layers=cfg.layers,
                                shard_files=shard_files,
                                deduplicate=cfg.deduplicate,
                                chunk=cfg.chunk,
                                chunk_size=cfg.chunk_size,
                                output_format=output_format,
                            )
                    total_files += waveform.shape[0]
                    total_batches += 1
                except (OSError, RuntimeError) as e:
                    if "CUDA out of memory" in str(e) and bz > 1:
                        logger.warning("CUDA OOM with batch size %d, reducing to %d", bz, 1)
                        oom_retry = True
                        break
                    logger.exception("An unexpected error occurred on waveform shape: %s", waveform.shape)
                    has_file_errors = True
                    with (cfg.output_folder / "files_with_error.txt").open("a") as f:
                        f.writelines(f"{path}\n" for path in file_paths)
            if not oom_retry:
                break
            bz = 1

    if output_type in ["codes", "embs_clustered_kmeans"]:
        for fout in shard_files.values():
            if fout:
                fout.close()

    return total_files, total_batches, has_file_errors


def merge_files(
    output_folder: Path,
    layer: int,
    split: str,
    dataset_name: str,
    n_workers: int,
) -> Path:
    output_file = (output_folder / str(layer) / split / dataset_name).with_suffix(".jsonl")
    with output_file.open("w") as full:
        for worker_id in range(n_workers):
            partial_path = worker_shard_path(output_folder, layer, split, dataset_name, worker_id, "jsonl")
            partial = partial_path.open()
            for line in partial:
                print(line.strip(), file=full)
            partial_path.unlink()

    return output_file


def _finalize_jsonl_manifest(
    cfg: FeatExtractConfig,
    context: DistributedContext,
    manifest: Path,
    manifest_split: str,
) -> None:
    """Merge rank-local JSONL shards and create benchmark pairs when needed."""
    if not context.is_leader or manifest_split not in {"test", "unit_vocoder"}:
        return

    for layer in cfg.effective_output_layers:
        flat_file = merge_files(
            output_folder=cfg.output_folder,
            layer=layer,
            split=manifest_split,
            dataset_name=manifest.stem,
            n_workers=context.world_size,
        )
        if manifest_split == "test":
            create_paired_file(flat_file)


@timing
def save_units(
    cfg: FeatExtractConfig,
    context: DistributedContext,
    output_type: OutputType,
    kmeans_path: Path | None = None,
) -> None:
    # context = init_distributed_context(cfg.distributed_port)
    logger.info("Distributed context %s", context)

    n_gpus = torch.cuda.device_count()

    for manifest_split in ["train", "valid", "test", "unit_vocoder"]:
        for manifest in getattr(cfg, f"manifests_{manifest_split}", []):
            logger.info("Extracting features for %s manifest %s", manifest_split, manifest)
            start = time.time()
            with _device_scope(context, n_gpus):
                batch_size = cfg.batch_size
                total_files, total_batches, _has_file_errors = transcribe_manifest(
                    cfg,
                    batch_size=batch_size,
                    manifest=manifest,
                    manifest_split=manifest_split,
                    rank=context.rank,
                    world_size=context.world_size,
                    output_type=output_type,
                    kmeans_path=kmeans_path,
                )

            total = time.time() - start
            logger.info(
                "Rank %s: time spent for %d files - %d batches = %s",
                context.local_rank,
                total_files,
                total_batches,
                total,
            )

            if context.world_size > 1:
                distr.barrier()

            _finalize_jsonl_manifest(cfg, context, manifest, manifest_split)


@timing
def save_embeddings_to_train_kmeans(cfg: FeatExtractConfig, context: DistributedContext) -> None:
    logger.info("Distributed context %s", context)

    n_gpus = torch.cuda.device_count()

    for manifest in cfg.manifests_train_kmeans:
        logger.info("Extracting features for manifest %s", manifest)
        with _device_scope(context, n_gpus):
            batch_size = cfg.batch_size
            _total_files, _total_batches, _has_file_errors = transcribe_manifest(
                cfg,
                batch_size=batch_size,
                manifest=manifest,
                manifest_split="train_kmeans",
                rank=context.rank,
                world_size=context.world_size,
                output_type="embs",
            )
        if context.world_size > 1:
            distr.barrier()


def run_feature_extraction_speech(cfg_feat_ext: FeatExtractConfig, context: DistributedContext) -> None:
    if context.is_leader:
        logger.info("Starting feature extraction")
        logger.info("Common folder: %s", cfg_feat_ext.output_folder)
        logger.info("cfg_feat_ext: %s", cfg_feat_ext)
        logger.info("cfg_feat_ext.cfg_kmeans: %s", cfg_feat_ext.cfg_kmeans)

    if cfg_feat_ext.clustering == "codes":
        save_units(cfg_feat_ext, context, "codes")
    elif cfg_feat_ext.clustering == "kmeans":
        kmeans_path = cfg_feat_ext.cfg_kmeans.km_path
        if kmeans_path is not None:
            if not kmeans_path.is_file():
                raise FileNotFoundError(f"pretrained k-means model not found: {kmeans_path}")
            if context.is_leader:
                logger.info("Using pretrained k-means model: %s", kmeans_path)
        else:
            save_embeddings_to_train_kmeans(cfg_feat_ext, context)

            layer_of_interest = str(cfg_feat_ext.layers[0])
            kmeans_path = cfg_feat_ext.output_folder / layer_of_interest / "kmeans_model" / "kmeans.joblib"
            kmeans_path.parent.mkdir(parents=True, exist_ok=True)
            if context.is_leader:
                # Train kmeans on a single CPU (using scikit on CPU for now)
                learn_kmeans(
                    cfg=cfg_feat_ext.cfg_kmeans,
                    embs_folder_path=cfg_feat_ext.output_folder / layer_of_interest / "train_kmeans",
                    model_save_path=kmeans_path,
                )

            if context.world_size > 1:
                distr.barrier()

        save_units(cfg_feat_ext, context, output_type="embs_clustered_kmeans", kmeans_path=kmeans_path)
