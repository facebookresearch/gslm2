import argparse
import os
from pathlib import Path

from gslm2.encoder.config import SpeechConfig, load_config
from gslm2.shared.slurm import (
    default_train_job_config,
    launch_with_submitit,
    merge_slurm_config,
    slurm_config_parse_args,
)
from gslm2.shared.tools import init_logger


def main() -> None:
    cluster = default_train_job_config(os.getenv("GSLM2_CLUSTER"))
    parser = argparse.ArgumentParser(
        description="Launch native speech-encoder training runs.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("configs", type=Path, nargs="+", help="Config file(s) for the training run(s).")
    parser.add_argument("-A", "--account", type=str, help="Slurm account")
    parser.add_argument("-N", "--nodes", type=int, help="Number of nodes")
    parser.add_argument("-G", "--gpus-per-node", type=int, help="GPUs per node")
    parser.add_argument("-c", "--cpus-per-task", type=int, help="CPUs per task", default=cluster.get("cpus_per_task"))
    parser.add_argument("--mem-per-gpu", type=str, help="Memory per GPU", default=cluster.get("mem_per_gpu"))
    parser.add_argument("-t", "--time", type=int, help="Time limit in minutes", default=cluster.get("time"))
    parser.add_argument("-C", "--constraint", type=str, help="Slurm constraint", default=cluster.get("constraint"))
    parser.add_argument("-q", "--qos", type=str, help="Slurm QoS")
    parser.add_argument("-p", "--partition", type=str, help="Slurm partition")
    parser.add_argument("-x", "--exclude", type=str, help="Slurm nodes to exclude")
    parser.add_argument("--dump", type=Path, default=os.getenv("GSLM2_SUBMITIT_FOLDER"), help="Submitit dump")
    args = parser.parse_args()

    init_logger()
    if args.dump is None:
        parser.error("Dump directory must be specified with --dump or by setting $GSLM2_SUBMITIT_FOLDER")

    def lazy_train(cfg: SpeechConfig) -> None:
        from gslm2.encoder.training.trainer import train

        train(cfg)

    jobs = [(lazy_train, (load_config(cfg),)) for cfg in args.configs]
    job_name = jobs[0][1][0].run.dir.name
    slurm = merge_slurm_config(jobs[0][1][0].run.slurm, slurm_config_parse_args(args))

    # --dump is always the final run directory (both initial launch and resume).
    # Pass dump_subfolder="" so launch_with_submitit uses it as-is without nesting.
    launch_with_submitit(
        job_name,
        jobs,
        args.dump,
        slurm,
        copy_code=True,
        dump_subfolder="",
    )


if __name__ == "__main__":
    main()
