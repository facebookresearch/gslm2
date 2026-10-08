import contextvars
import logging
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch._appdirs
import torch._inductor.config
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from gslm2.tools import init_logger

logger = logging.getLogger()

_mesh = contextvars.ContextVar[DeviceMesh]("mesh")

_EVAL_NCCL_TIMEOUT = timedelta(minutes=5)


def restrict_cuda_to_local_rank() -> None:
    """Restrict this process's CUDA visibility to a single GPU by local rank.

    Must run before the first ``torch.cuda`` call, so PyTorch's lazy CUDA
    initialisation picks up the narrowed ``CUDA_VISIBLE_DEVICES``. Importing
    torch is fine — it does not initialise CUDA. Afterwards each rank sees
    exactly one device as ``cuda:0`` and NCCL P2P works via cudaIpc handles
    between sibling processes on the same node.

    We do this ourselves rather than asking SLURM for cgroup-level isolation via
    ``--gpus-per-task=1``, which breaks NCCL P2P with
    ``Cuda failure 101 'invalid device ordinal'``.

    When ``CUDA_VISIBLE_DEVICES`` already lists the allocation's physical
    device IDs (e.g. SLURM sets ``4,5`` for a 2-GPU allocation), the local rank
    indexes *into* that list rather than replacing it — writing the bare local
    rank would bind rank 0 to physical GPU 0, outside the allocation. A local
    rank beyond the end of the list is a misconfiguration, so we fail loudly
    instead of silently grabbing a device we were not given.

    No-op when the local rank cannot be determined, or when
    ``CUDA_VISIBLE_DEVICES`` already names a single device (i.e. something else
    has already narrowed it).
    """
    local = os.environ.get("SLURM_LOCALID") or os.environ.get("LOCAL_RANK")
    if local is None:
        return
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current is not None and "," not in current:
        return
    if current:
        visible = [tok.strip() for tok in current.split(",") if tok.strip()]
        index = int(local)
        if index >= len(visible):
            raise ValueError(
                f"local rank {index} has no device in CUDA_VISIBLE_DEVICES={current!r} "
                f"({len(visible)} device(s) allocated); refusing to bind outside the allocation"
            )
        selected = visible[index]
    else:
        selected = local
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    logger.debug("Restricted CUDA_VISIBLE_DEVICES to %s (local rank %s of %s)", selected, local, current)


class DistributedNotInitializedError(RuntimeError):
    """Raised when trying to access distributed variables before initializing the distributed environment."""


def pg_ddp() -> dist.ProcessGroup:
    try:
        return _mesh.get().get_group("ddp")
    except LookupError as error:
        raise DistributedNotInitializedError from error


def pg_metalearning() -> dist.ProcessGroup:
    try:
        return _mesh.get().get_group("metalearning")
    except LookupError as error:
        raise DistributedNotInitializedError from error


@dataclass(frozen=True)
class DistributedEnvironment:
    global_rank: int
    local_rank: int
    world_size: int
    master_port: int
    master_addr: str


def distributed_environment(*, min_master_port: int = 20_000, max_master_port: int = 60_000) -> DistributedEnvironment:
    if os.getenv("LOCAL_RANK") is not None:  # torchrun job
        env = DistributedEnvironment(
            global_rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            master_port=int(os.environ["MASTER_PORT"]),
            master_addr=os.environ["MASTER_ADDR"],
        )
    elif os.getenv("SLURM_JOB_ID") is not None:  # Slurm job
        scontrol_cmd = ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        master_addr = subprocess.check_output(scontrol_cmd, text=True).split()[0]
        # Mix in SLURM_RESTART_COUNT so requeued jobs try a fresh port instead of
        # repeatedly colliding with a stale process holding the previous one.
        port_seed = f"{os.environ['SLURM_JOB_ID']}:{os.environ.get('SLURM_RESTART_COUNT', '0')}"
        master_port = random.Random(port_seed).randint(min_master_port, max_master_port)
        env = DistributedEnvironment(
            global_rank=int(os.environ["SLURM_PROCID"]),
            local_rank=int(os.environ["SLURM_LOCALID"]),
            world_size=int(os.environ["SLURM_NTASKS"]),
            master_port=master_port,
            master_addr=master_addr,
        )
    else:
        env = DistributedEnvironment(  # Single GPU job
            global_rank=0,
            local_rank=0,
            world_size=1,
            master_port=random.Random(-1).randint(min_master_port, max_master_port),
            master_addr="127.0.0.1",
        )
    os.environ["RANK"] = str(env.global_rank)
    os.environ["LOCAL_RANK"] = str(env.local_rank)
    os.environ["WORLD_SIZE"] = str(env.world_size)
    os.environ["MASTER_PORT"] = str(env.master_port)
    os.environ["MASTER_ADDR"] = env.master_addr
    return env


def device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _init_process_group(*, timeout: timedelta | None = None, device_id: torch.device | None = None) -> None:
    """Initialize the default process group if it isn't already. Idempotent."""
    if dist.is_initialized():
        return
    kwargs: dict[str, object] = {"backend": "nccl" if torch.cuda.is_available() else "gloo"}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if device_id is not None:
        kwargs["device_id"] = device_id
    dist.init_process_group(**kwargs)  # type: ignore[arg-type]


def setup_distributed(*, num_workers: int = 1) -> None:
    """Initialize the process group and a 2-D ``(ddp, metalearning)`` device mesh.

    Used by the trainers. Evaluations want :func:`setup_ddp` instead, which
    skips the mesh and narrows CUDA visibility per rank.
    """
    env = distributed_environment()
    logger.debug("World size %s, local rank %s, global rank %s", env.world_size, env.local_rank, env.global_rank)
    mesh_shape = (env.world_size // num_workers, num_workers)
    if env.world_size % num_workers != 0:
        raise ValueError(f"Invalid number of workers ({num_workers}): must divide world size {env.world_size}")
    _init_process_group()
    _mesh.set(init_device_mesh(device_type(), mesh_shape, mesh_dim_names=("ddp", "metalearning")))
    logger.debug("DeviceMesh initialized with shape %s", mesh_shape)


def _flush_print(msg: str) -> None:
    """Print with immediate flush — survives C-level hangs that swallow logger output."""
    print(msg, flush=True)  # noqa: T201
    sys.stdout.flush()


def setup_ddp() -> tuple[int, int, int, bool]:
    """Initialize plain DDP for evaluation jobs. Idempotent.

    Unlike :func:`setup_distributed` this builds no device mesh, restricts each
    rank's CUDA visibility to one GPU (see
    :func:`restrict_cuda_to_local_rank`), and applies a
    bounded NCCL timeout so a stuck rank fails instead of hanging forever.

    Returns:
        ``(global_rank, local_rank, world_size, is_distributed)``.
    """
    env = distributed_environment()
    is_distributed = env.world_size > 1
    if is_distributed and dist.is_available() and dist.is_initialized():
        return env.global_rank, env.local_rank, env.world_size, True

    restrict_cuda_to_local_rank()
    logger.info(
        "[setup_ddp] rank=%d local_rank=%d world_size=%d cuda_visible=%s device_count=%d master=%s:%s",
        env.global_rank,
        env.local_rank,
        env.world_size,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        torch.cuda.device_count(),
        env.master_addr,
        env.master_port,
    )

    if torch.cuda.is_available():
        # Each rank sees exactly one device as cuda:0 (CVD narrowed above).
        torch.cuda.set_device(0)

    if not is_distributed:
        logger.info("Running in single-process mode (world_size=1)")
        return env.global_rank, env.local_rank, env.world_size, False

    _flush_print(f"[setup_ddp] rank={env.global_rank} calling init_process_group ...")
    _init_process_group(
        timeout=_EVAL_NCCL_TIMEOUT,
        device_id=torch.device("cuda:0") if torch.cuda.is_available() else None,
    )
    _flush_print(f"[setup_ddp] rank={env.global_rank} init_process_group done, calling barrier ...")
    dist.barrier()
    _flush_print(f"[setup_ddp] rank={env.global_rank} barrier done")
    logger.info(
        "DDP initialized: rank=%d, local_rank=%d, world_size=%d",
        env.global_rank,
        env.local_rank,
        env.world_size,
    )
    return env.global_rank, env.local_rank, env.world_size, True


def cleanup_ddp() -> None:
    """Destroy the default process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed")


def is_dist_initialized() -> bool:
    """Whether torch.distributed is available and the process group is up."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """World size, or 1 when not running distributed."""
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank() -> int:
    """Global rank, or 0 when not running distributed."""
    return dist.get_rank() if is_dist_initialized() else 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def setup_pytorch(*, use_deterministic: bool) -> None:
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cudnn.benchmark = False  # Needed because of dynamic input size
    torch.autograd.set_detect_anomaly(mode=False, check_nan=True)
    if use_deterministic:
        torch.use_deterministic_algorithms(mode=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch._inductor.config.fallback_random = True  # noqa: SLF001  no public inductor knob


def setup_environment(**kwargs: str) -> None:
    if "SLURM_JOB_ID" in os.environ and (Path("/scratch") / os.environ["SLURM_JOB_ID"]).is_dir():
        cache = Path("/scratch") / os.environ["SLURM_JOB_ID"]
        kwargs.setdefault("TMPDIR", str(cache / "tmp"))
    else:
        cache = Path(torch._appdirs.user_cache_dir(appname="spidr"))  # noqa: SLF001  no public cache-dir API
        kwargs.setdefault("TMPDIR", "/tmp")  # noqa: S108  local fs; network fs causes multiprocessing/wandb socket errors
    cache.mkdir(exist_ok=True, parents=True)
    kwargs["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
    kwargs["TRITON_HOME"] = str(cache)

    # Strip proxy env vars early: SLURM jobs inherit *_PROXY settings that can
    # interfere with NCCL bootstrap/collectives, causing hangs.
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        if var in os.environ:
            logger.debug("Removing inherited proxy env var: %s", var)
            del os.environ[var]

    for name, value in kwargs.items():
        if os.getenv(name) != str(value):
            os.environ[name] = str(value)
            logger.debug("Setting env: %s=%s", name, value)


def setup_training(seed: int, *, use_deterministic: bool = False, num_workers: int = 1) -> None:
    init_logger()
    logger.info("Starting training")
    setup_distributed(num_workers=num_workers)
    set_seed(seed + dist.get_rank())
    setup_pytorch(use_deterministic=use_deterministic)
    setup_environment()
