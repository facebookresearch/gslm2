"""Launch Slurm jobs with submitit."""

import argparse
import importlib.resources
import json
import logging
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import submitit
from submitit.core.utils import FailedJobError

from gslm2.config import SlurmConfig
from gslm2.tools import write_git_info_if_available

logger = logging.getLogger()

DEFAULT_VALIDATION_QOS = "h200_comm_shared"


def current_codebase_path() -> Path:
    import gslm2

    return Path(gslm2.__file__).parent.parent


def copy_code_to_submitit_folder(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(importlib.resources.files(__package__)).parent.parent
    logger.info("Copying: %s to: %s", root, output_dir)
    # Optional per-user excludes: one rsync pattern per line in ~/.metasr_rsync_excludes
    user_excludes = Path.home() / ".metasr_rsync_excludes"
    exclude_from = f"--exclude-from='{user_excludes}' " if user_excludes.is_file() else ""
    rsync_cmd = (
        f"rsync -ar --copy-links "
        f"--exclude='.git/' "
        f"--exclude='.git' "
        f"--exclude './data/' "
        f"--exclude './paper/' "
        f"--exclude __pycache__ "
        f"--exclude '*.pyc' "
        f"--exclude '*.ipynb' "
        f"--exclude '*.err' "
        f"--exclude '*.out' "
        f"--exclude '*.log' "
        f"--exclude '*.pt' "
        f"--exclude '.ruff_cache/' "
        f"--exclude '.mypy_cache/' "
        f"--exclude='.github/' "
        f"--exclude='.venv/' "
        f"--exclude='.pixi/' "
        f"--exclude='.pytest_cache/' "
        f"{exclude_from}"
        f"{root}/ {output_dir}"
    )
    subprocess.call([rsync_cmd], shell=True)
    logger.info("Copy done.")


def make_checkpointable[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    def checkpoint(*args, **kwargs) -> submitit.helpers.DelayedSubmission:  # noqa: ANN002, ANN003
        return submitit.helpers.DelayedSubmission(fn, *args, **kwargs)

    fn.checkpoint = checkpoint  # ty: ignore[unresolved-attribute]
    return fn


ClusterName = Literal["JZ_V100", "JZ_A100", "JZ_H100", "FAIR_AWS_A100", "FAIR_SC", "FAIR_SC_3"]


def default_train_job_config(cluster_name: ClusterName | None) -> dict:
    match cluster_name:
        case None:
            logger.warning(
                "Env. variable cluster name 'GSLM2_CLUSTER' is None, using default Slurm parameters (max time = 5min)"
            )
            return {}
        case "JZ_V100":
            return {"time": 1200, "cpus_per_task": 10, "mem_per_gpu": None, "constraint": None}
        case "JZ_A100":
            return {"time": 1200, "cpus_per_task": 8, "mem_per_gpu": None, "constraint": "a100"}
        case "JZ_H100":
            return {"time": 1200, "cpus_per_task": 24, "mem_per_gpu": None, "constraint": "h100"}
        case "FAIR_AWS_A100":
            return {"time": 4320, "cpus_per_task": 12, "mem_per_gpu": "80g", "constraint": None}
        case "FAIR_SC" | "FAIR_SC_3":
            return {"time": 4320, "cpus_per_task": 12, "mem_per_gpu": "120g", "constraint": None}
    raise ValueError(f"Unknown cluster '{cluster_name}'")


def validation_job_config() -> SlurmConfig:
    try:
        account, cpus_per_task = os.environ["SLURM_JOB_ACCOUNT"], os.environ["SLURM_CPUS_PER_TASK"]
    except KeyError as error:
        raise ValueError("This function should only be used inside a SLURM job.") from error
    return SlurmConfig(
        account=account, nodes=1, gpus_per_node=8, time=480, cpus_per_task=cpus_per_task, qos=DEFAULT_VALIDATION_QOS
    )


def slurm_config_parse_args(args: argparse.Namespace) -> SlurmConfig:
    """Parse CLI args into a SlurmConfig, skipping None values."""
    config = SlurmConfig(
        {key: getattr(args, key) for key in SlurmConfig.__required_keys__ if getattr(args, key) is not None}
    )
    for key in SlurmConfig.__optional_keys__:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
    return config


def merge_slurm_config(from_yaml: SlurmConfig | None, from_cli: SlurmConfig) -> SlurmConfig:
    """Merge SLURM config from YAML and CLI args. CLI args take precedence,
    except for ``qos`` which is taken from the YAML when set (per-config QoS
    is an explicit choice and should override the launcher default).
    """
    base = dict(from_yaml) if from_yaml else {}
    base.update(from_cli)
    if from_yaml and from_yaml.get("qos"):
        base["qos"] = from_yaml["qos"]
    missing = SlurmConfig.__required_keys__ - base.keys()
    if missing:
        raise SystemExit(
            f"Missing required SLURM config fields: {', '.join(sorted(missing))}. "
            "Provide them in the YAML config under run.slurm or via CLI args (-A, -N, -G)."
        )
    return SlurmConfig(**base)


@submitit.helpers.clean_env()
def launch_with_submitit(  # noqa: PLR0913
    job_name: str,
    jobs: list[tuple[Callable, tuple]],
    dump: str | Path,
    slurm: SlurmConfig,
    *,
    copy_code: bool = True,
    dump_subfolder: str | None = None,
) -> bool:
    launch_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    subfolder = dump_subfolder if dump_subfolder is not None else job_name
    folder = Path(dump) / subfolder if subfolder else Path(dump)
    # Only the default job-name folder is uniquified; an explicit dump_subfolder is reused as given
    # so that resuming into an existing run directory works.
    if dump_subfolder is None and folder.exists():
        folder = Path(dump) / f"{job_name}_{launch_time}"
    folder.mkdir(parents=True, exist_ok=True)

    if copy_code:
        copy_code_to_submitit_folder(folder / "code")
        codebase_path = folder / "code" / "src"
    else:  # No-op in most situations, but correctly handles nested cases
        codebase_path = current_codebase_path()

    write_git_info_if_available(folder)
    job_info = {"job_name": job_name, "launch_time": launch_time} | slurm
    (folder / "job-info.json").write_text(json.dumps(job_info, indent=2))

    os.chdir(codebase_path)
    logger.info("Launching from %s", codebase_path)
    executor = submitit.SlurmExecutor(folder=folder / "%j")
    executor.update_parameters(job_name=job_name, ntasks_per_node=slurm["gpus_per_node"], **slurm)
    try:
        with executor.batch():
            for fn, args in jobs:
                executor.submit(make_checkpointable(fn), *args)
    except FailedJobError as error:
        logger.warning("Didn't managed to submit %s: %s", job_name, error)
        return False
    logger.info("Submitted %s (%s jobs)", job_name, len(jobs))
    return True
