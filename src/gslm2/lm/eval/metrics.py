# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

# Always call setup_fairseq2() before using any fairseq2 functionality
from fairseq2.device import Device
from fairseq2.gang import Gang
from fairseq2.metrics import MetricBag, sync_and_compute_metrics
from fairseq2.metrics.aggregation import Mean
from fairseq2.nn import BatchLayout
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from tqdm import tqdm

from gslm2.shared.distributed import DistributedContext
from gslm2.shared.utils.typing import PathLike

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def apply_custom_tokeniser(x: list[int], offset: int, bos: int | None) -> list[int]:
    """Add bos token and offset all values by specified offset"""
    if bos is None:
        return [val + offset for val in x]
    return [bos] + [val + offset for val in x]


class UnitPairsDataset(Dataset):
    def __init__(self, jsonl_path: Path, offset: int, bos: int | None) -> None:
        super().__init__()
        self.offset = offset
        self.bos = bos
        with jsonl_path.open("r") as file:
            self.data = [json.loads(line) for line in file]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], str, str]:
        """Get an item by index"""
        return self.get_item_by_idx(idx)

    def get_item_by_idx(self, idx: int) -> tuple[list[int], list[int], str, str]:
        """Get an item by index"""
        x = self.data[idx]

        pos_units = x["positive"]["units"]
        neg_units = x["negative"]["units"]
        # Multi-stream manifests use "|" to delimit per-stream unit sequences.
        if "|" in pos_units and "|" in neg_units:
            pos = []
            neg = []
            for _pos, _neg in zip(pos_units.split("|"), neg_units.split("|"), strict=True):
                _pos = list(map(int, _pos.split()))
                _neg = list(map(int, _neg.split()))
                pos.append(apply_custom_tokeniser(_pos, self.offset, self.bos))
                neg.append(apply_custom_tokeniser(_neg, self.offset, self.bos))
        else:
            pos = list(map(int, pos_units.split()))
            neg = list(map(int, neg_units.split()))

            pos, neg = (
                apply_custom_tokeniser(pos, self.offset, self.bos),
                apply_custom_tokeniser(neg, self.offset, self.bos),
            )

        pos_file = Path(x["positive"]["file_name"]).stem
        neg_file = Path(x["negative"]["file_name"]).stem
        return pos, neg, pos_file, neg_file


def collate_fn(
    batch: list[tuple[list[int], list[int], str, str]],
) -> tuple[list[list[int]], list[list[int]], list[str], list[str]]:
    """Custom collate function for the DataLoader"""
    pos_samples = [item[0] for item in batch]
    neg_samples = [item[1] for item in batch]
    pos_files = [item[2] for item in batch]
    neg_files = [item[3] for item in batch]
    return pos_samples, neg_samples, pos_files, neg_files


class SLMEvalMetrics:
    """Holds the metrics of a sequence model training or evaluation task."""

    names: list[str]
    paths: list[PathLike]
    bag: MetricBag
    gang: Gang

    def __init__(self, metric_paths: Sequence[PathLike], gang: Gang) -> None:
        device = gang.device
        self.bag = MetricBag(device)

        self.names = SLMEvalMetrics.prepare_metric_bag(metric_paths, self.bag)

        self.gang = gang
        self.paths = list(metric_paths)

    @staticmethod
    def prepare_metric_bag(metric_paths: Sequence[PathLike], bag: MetricBag) -> list[str]:
        names = [Path(path).stem for path in metric_paths]

        for metric_name in names:
            bag.add(f"{metric_name}_accuracy", Mean())
            bag.add(f"{metric_name}_accuracy_per_token", Mean())
            # This is not super useful, clutters logs and wandb
            # bag.add(f"{metric_name}_npairs", Sum())

        return names

    @torch.inference_mode()
    def _update_accuracy(self, metric_name: str, batch_results: list[bool], *, is_normalized: bool) -> None:
        if is_normalized:
            metric = self.bag.metrics[f"{metric_name}_accuracy_per_token"]
        else:
            metric = self.bag.metrics[f"{metric_name}_accuracy"]
        metric.update(float(np.mean(batch_results)), weight=len(batch_results))

    def update_accuracy(self, metric_name: str, batch_results: list[bool]) -> None:
        """Update accuracy for a batch of results"""
        self._update_accuracy(metric_name, batch_results, is_normalized=False)

    def update_accuracy_per_token(self, metric_name: str, batch_results: list[bool]) -> None:
        """Update accuracy per token for a batch of results"""
        self._update_accuracy(metric_name, batch_results, is_normalized=True)

    @torch.inference_mode()
    def update_npairs(self, metric_name: str, npairs: int) -> None:
        """Update the number of pairs for a metric"""
        # Turned into a no-op, if we need it again, uncomment also in prepare_metric_bag
        # self.bag.metrics[f"{metric_name}_npairs"].update(npairs)

    def sync_and_compute_metrics(self) -> dict[str, object] | None:
        """Sync and compute metrics across all processes"""
        return sync_and_compute_metrics(self.bag, self.gang)


def calc_nll(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    calculate the negative log likelihood of the logits given the target
    :param logits: logits
    :param target: target
    :param mask: mask
    :return: nll, nll_norm
    """
    # Calculate the cross-entropy loss for each sequence
    losses = cross_entropy(
        logits.contiguous().view(-1, logits.size(-1)), target.contiguous().view(-1), reduction="none"
    )

    # Reshape the losses to match the original sequences
    losses = losses.view(*target.size())

    # Use the mask to ignore the losses of the padding tokens
    masked_losses = losses * mask

    # Sum the losses to get the total loss for each sequence
    nll = masked_losses.sum(dim=-1)
    nll_norm = nll / mask.sum(dim=-1)

    return nll, nll_norm


def prepare_batch(sequences: list[list[int]], device: Device) -> tuple[torch.Tensor, list[int]]:
    """
    Prepare a batch of sequences for the model by padding and converting to tensors
    Optimized version to reduce CPU-GPU transfers and tensor creation overhead
    """
    # Find the max length for padding
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    # Create tensors all at once on CPU, then transfer to GPU
    padded = torch.zeros((batch_size, max_len), dtype=torch.long)
    seq_lens = []

    # Fill padded tensor with actual sequences
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        seq_lens.append(seq_len)
        padded[i, :seq_len] = torch.tensor(seq, dtype=torch.long)

    # Transfer to device in one operation
    return padded.to(device), seq_lens


@torch.inference_mode()
def log_likelihood_fs2(
    model: torch.nn.Module, sequences: list[list[int]], device: Device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate log likelihood for a batch of sequences
    """
    if isinstance(model.decoder_frontend, torch.nn.ModuleList):
        return log_likelihood_fs2_multistream(model, sequences, device)

    seqs, seq_lens = prepare_batch(sequences, device)
    seqs_layout = BatchLayout(shape=seqs.shape, seq_lens=seq_lens, packed=False, device=device)

    logits = model(seqs, seqs_layout)

    shifted_x = seqs[..., 1:]
    shifted_logits = logits[..., :-1, :]

    # Create mask based on sequence lengths (accounting for the shift)
    max_len = shifted_x.shape[1]
    # Pre-compute lengths tensor once and keep on device
    lengths_tensor = torch.tensor(seq_lens, device=device) - 1
    # Create arange once and reuse
    arange_tensor = torch.arange(max_len, device=device).unsqueeze(0)
    # Create mask directly as long tensor to avoid type conversion
    mask = (arange_tensor < lengths_tensor.unsqueeze(1)).long()

    # Calculate the likelihoods
    nll, nll_norm = calc_nll(shifted_logits, shifted_x, mask)
    return -nll, -nll_norm


def prepare_batch_multistream(sequences: list[list[list[int]]], device: Device) -> tuple[torch.Tensor, list[int]]:
    """Prepare a batch of multi-stream sequences for the model.

    Each sequence in ``sequences`` is itself a list of ``S`` per-stream lists of ints.
    All streams within a single sequence are truncated to the minimum stream length
    (so they remain aligned), then the streams are interleaved into a single
    ``(B*S, T)`` tensor matching the layout expected by
    :class:`~gslm2.lm.models.opt.multi_stream_model.TransformerLMMultiStream`.
    """
    seqs = []
    seq_lens = []

    for seq in sequences:
        min_len = min(len(s) for s in seq)
        seqs.append(torch.stack([torch.tensor(s[:min_len], dtype=torch.long) for s in seq], dim=0))
        seq_lens.extend(min_len for _ in seq)

    seqs_pad = []
    max_len = max(seq_lens)
    for seq in seqs:
        pad_size = (0, max_len - seq.size(1))
        seqs_pad.append(torch.nn.functional.pad(seq, pad_size, value=0))
    seqs_pad = torch.stack(seqs_pad, dim=0).reshape(-1, max_len)

    return seqs_pad.to(device), seq_lens


@torch.inference_mode()
def log_likelihood_fs2_multistream(
    model: torch.nn.Module, sequences: list[list[list[int]]], device: Device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pair log-likelihoods on a multi-stream model.

    Returns the average (across streams) log-likelihood (un-normalised and
    per-token) for each input sequence.
    """
    seqs, seq_lens = prepare_batch_multistream(sequences, device)
    seqs_layout = BatchLayout(shape=seqs.shape, seq_lens=seq_lens, packed=False, device=device)

    num_streams = len(model.decoder_frontend)
    logits_list = model(seqs, seqs_layout)

    total_nll, total_nll_norm = 0.0, 0.0
    for i, logits in enumerate(logits_list):
        seqs_stream = seqs[i::num_streams]
        shifted_x = seqs_stream[:, 1:]
        shifted_logits = logits[..., :-1, :]

        # Create mask based on sequence lengths (accounting for the shift)
        max_len = shifted_x.shape[1]
        # Pre-compute lengths tensor once and keep on device
        lengths_tensor = torch.tensor(seq_lens[i::num_streams], device=device) - 1
        # Create arange once and reuse
        arange_tensor = torch.arange(max_len, device=device).unsqueeze(0)
        # Create mask directly as long tensor to avoid type conversion
        mask = (arange_tensor < lengths_tensor.unsqueeze(1)).long()

        change_mask = (shifted_x != seqs_stream[:, :-1]).long()
        candidate_mask = mask * change_mask
        has_changes = candidate_mask.sum(dim=-1, keepdim=True) > 0
        mask = torch.where(has_changes, candidate_mask, mask)

        # Calculate the likelihoods
        nll, nll_norm = calc_nll(shifted_logits, shifted_x, mask)

        total_nll += -nll
        total_nll_norm += -nll_norm

    total_nll /= num_streams
    total_nll_norm /= num_streams

    return total_nll, total_nll_norm


def _initialize_eval_process_group(local_rank: int, rank: int, world_size: int) -> None:
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Initialize distributed process group
    dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized() and world_size > 1:
        try:
            # Try to initialize with environment variables
            dist.init_process_group(backend=dist_backend)
            logger.info("Initialized process group with env variables")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to initialize with env variables: %s", e)
            # Fallback to explicit initialization
            dist.init_process_group(backend=dist_backend, rank=rank, world_size=world_size)


def setup_distributed(args_local_rank: int = -1) -> DistributedContext:
    """
    Initialize the distributed environment

    Supports multiple distributed training setups:
    1. Command line local_rank (torch.distributed.launch)
    2. SLURM environment variables (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID)
    3. Standard PyTorch distributed environment variables (RANK, WORLD_SIZE, LOCAL_RANK)
    4. Single GPU fallback

    Args:
        args_local_rank: Local rank passed from command line arguments
    """

    if args_local_rank >= 0:
        # Use the local_rank provided by command line
        local_rank = args_local_rank
        # When using torch.distributed.launch, these should be set
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    elif all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        # SLURM environment variables
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        # Try to get local rank from SLURM_LOCALID first, then fallback to other methods
        if "SLURM_LOCALID" in os.environ:
            local_rank = int(os.environ["SLURM_LOCALID"])
        elif "SLURM_NODEID" in os.environ and "SLURM_NTASKS_PER_NODE" in os.environ:
            # Calculate local rank based on node ID and tasks per node
            node_id = int(os.environ["SLURM_NODEID"])
            ntasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
            local_rank = rank - (node_id * ntasks_per_node)
        else:
            # Fallback: assume tasks are distributed evenly across available GPUs
            local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Set environment variables for compatibility with other distributed libraries
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        # Optional: Set master address and port if not already set
        if "MASTER_ADDR" not in os.environ:
            # Try to use SLURM_LAUNCH_NODE_IPADDR or fallback to localhost
            master_addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
            os.environ["MASTER_ADDR"] = master_addr
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"  # Default PyTorch port

        logger.info("Using SLURM environment: PROCID=%d, NTASKS=%d, LOCALID=%d", rank, world_size, local_rank)
        logger.info(
            "SLURM MASTER_ADDR=%s, MASTER_PORT=%s", os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT")
        )
    else:
        logger.info("Distributed environment variables not set; using one process")
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

    _initialize_eval_process_group(local_rank, rank, world_size)

    # Get updated rank and world_size after initialization
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    logger.info("Initialized process %d/%d with local rank %d", rank + 1, world_size, local_rank)

    return DistributedContext(
        is_distributed=dist.is_initialized(),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        mode="slurm" if "SLURM_PROCID" in os.environ else "launch" if args_local_rank >= 0 else "none",
    )


def cleanup_distributed() -> None:
    """
    Clean up the distributed environment
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def eval_lm(  # noqa: PLR0913
    model: torch.nn.Module,
    metrics: SLMEvalMetrics,
    gang: Gang,
    max_pairs: int | None = None,
    batch_size: int | None = 32,
    dataloader_num_workers: int = 4,
    seed: int = 0,
    offset: int = 2,
    bos: int | None = None,
) -> dict[str, object]:
    is_main_process = gang.rank == 0

    for metric_name, metric_path in zip(metrics.names, metrics.paths, strict=False):
        file_jsonl = Path(metric_path)

        if is_main_process:
            logger.info("Evaluating %s", metric_name)

        # Create dataset
        original_dataset = UnitPairsDataset(file_jsonl, offset=offset, bos=bos)

        # Select subset if needed
        if max_pairs is not None:
            # Use a deterministic subset based on seed
            rng = np.random.RandomState(seed)
            indices = rng.choice(
                len(original_dataset), size=min(len(original_dataset), int(max_pairs)), replace=False
            ).tolist()
            # Use PyTorch's Subset
            dataset = Subset(original_dataset, indices)
        else:
            dataset = original_dataset

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=gang.size,
            rank=gang.rank,
            shuffle=False,  # Keep deterministic ordering
            seed=seed,
        )

        # Create DataLoader with distributed sampler
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=dataloader_num_workers,
            pin_memory=True,
        )

        # Process in batches with progress bar only on main process
        for pos_tokens, neg_tokens, _, _ in tqdm(
            dataloader, desc=f"Processing {metric_name}", disable=not is_main_process
        ):
            # Process positive and negative samples
            pos_ll, pos_ll_norm = log_likelihood_fs2(model, pos_tokens, gang.device)
            neg_ll, neg_ll_norm = log_likelihood_fs2(model, neg_tokens, gang.device)

            # Keep operations on GPU as much as possible
            batch_results = (pos_ll > neg_ll).tolist()
            batch_results_norm = (pos_ll_norm > neg_ll_norm).tolist()

            metrics.bag.begin_updates()
            metrics.update_accuracy(metric_name, batch_results)
            metrics.update_accuracy_per_token(metric_name, batch_results_norm)
            metrics.update_npairs(metric_name, len(batch_results))
            metrics.bag.commit_updates()

    # Gather results from all processes
    logger.debug("Process %d: Entering sync_and_compute_metrics()", gang.rank)
    results = metrics.sync_and_compute_metrics()
    logger.debug("Process %d: Finished sync_and_compute_metrics()", gang.rank)
    assert results is not None

    return results
