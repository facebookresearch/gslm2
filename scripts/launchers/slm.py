import argparse
import copy
import getpass
import logging
import math
import os
import tempfile
import typing as tp
from pathlib import Path
from typing import get_args

from gslm2.eval.slm.config import deep_update
from gslm2.slurm import ClusterName, default_train_job_config, launch_with_submitit

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

ModelType = tp.Literal["spidr", "hubert", "wav2vec2", "wavlm", "text"]
DataSubset = str  # validated at parse time against DATA_SUBSETS
ModelSize = tp.Literal["125m", "350m", "1b", "3b", "7b"]
Clustering = tp.Literal["codes", "kmeans"]

BASE_CONFIG_PATH = "src/gslm2/eval/slm/base_configs/config_slm_base.yaml"
COPY_CODE = True


DATA_SUBSETS = {
    "600h": {
        "slm": {
            "feature_extraction": {
                "manifests_train": [
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/small_segmented_60s.csv"
                ],
            },
            "train": {
                "regime": {"num_steps": 15000},
            },
        }
    },
    "6kh": {
        "slm": {
            "feature_extraction": {
                "manifests_train": [
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/small_segmented_60s.csv",
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/medium_segmented_60s.csv",
                ],
            },
            "train": {
                "regime": {"num_steps": 25000},
            },
        }
    },
    "60kh": {
        "slm": {
            "feature_extraction": {
                "manifests_train": [
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/large_segmented_60s_train.csv",
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/medium_segmented_60s.csv",
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/small_segmented_60s.csv",
                ],
            },
            "train": {
                "regime": {"num_steps": 150000},
            },
        }
    },
    "combined_os_250k": {
        "slm": {
            "feature_extraction": {
                "manifests_train": [
                    "/datasets/metaspeechssl/shared/assets/manifests/emilia/emilia_en.csv",
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/large_segmented_60s_train_max900sec.csv",
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/medium_segmented_60s_max900sec.csv",
                    "/datasets/metaspeechssl/shared/assets/manifests/librilight/small_segmented_60s_max900sec.csv",
                    "/datasets/metaspeechssl/shared/assets/manifests/peoples_speech/postprocessed_all.csv",
                    "/datasets/metaspeechssl/shared/assets/manifests/voxpopuli_lid_filtered/segmented_60s_silsplit/full/en.csv",
                ],
            },
            "train": {
                "regime": {"num_steps": 250000},
            },
        }
    },
}

MODEL_SIZES = {
    "125m": {
        "nodes": 1,
        "config": {
            "slm": {
                "train": {
                    "model_preset": "opt125m",
                    "optimizer": {"config": {"lr": 1e-2}},
                    "dataset": {"max_num_tokens": 81920},
                },
            }
        },
    },
    "350m": {
        "nodes": 1,
        "config": {
            "slm": {
                "train": {
                    "model_preset": "opt350m",
                    "optimizer": {"config": {"lr": 5e-3}},
                    "dataset": {"max_num_tokens": 81920},
                },
            }
        },
    },
    "1b": {
        "nodes": 2,
        "config": {
            "slm": {
                "train": {
                    "model_preset": "opt1b",
                    "optimizer": {"config": {"lr": 1e-3}},
                    "dataset": {"max_num_tokens": 40960},
                },
            }
        },
    },
    "3b": {
        "nodes": 4,
        "config": {
            "slm": {
                "train": {
                    "model_preset": "opt3b",
                    "optimizer": {"config": {"lr": 5e-4}},
                    "dataset": {"max_num_tokens": 20480},
                },
            }
        },
    },
    "7b": {
        "nodes": 4,
        "config": {
            "slm": {
                "train": {
                    "model_preset": "opt7b",
                    "optimizer": {"config": {"lr": 2e-4}},
                    "dataset": {"max_num_tokens": 20480},
                },
            }
        },
    },
}

MODEL_CKPTS = {
    "spidr": "/datasets/metaspeechssl/shared/models/spidr_checkpoints/ssl_paper/spidr-librispeech-compile/step_400000.pt",
    "spidr_v2": "/datasets/metaspeechssl/shared/models/spidr_checkpoints/spidr-2.0/seed-0__final.pt",
}

# Maps SSL ckpt to model type
SSL_NAME_TO_MODEL_TYPE = {
    "spidr": "spidr",
    "spidr_v2": "spidr",
}


def update_data_subset(config: dict, subset: DataSubset, *, do_update_postfix: bool = True) -> dict:
    config = copy.deepcopy(config)
    update = copy.deepcopy(DATA_SUBSETS[subset])
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + "_" + subset
    return deep_update(config, update)


def update_model_size(config: dict, model_size: ModelSize, *, do_update_postfix: bool = True) -> dict:
    config = copy.deepcopy(config)
    update = copy.deepcopy(MODEL_SIZES[model_size]["config"])
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + "_opt-" + model_size
    return deep_update(config, update)


def get_model_size_nodes(model_size: ModelSize) -> int:
    return MODEL_SIZES[model_size]["nodes"]


def update_layer(config: dict, layers: list[int], *, do_update_postfix: bool = True) -> dict:
    """Set the SSL layer(s) used for feature extraction (a set for one job, not a sweep)."""
    config = copy.deepcopy(config)
    update = {"slm": {"feature_extraction": {"layers": list(layers)}}}
    if do_update_postfix:
        tag = f"layer-{layers[0]:02d}" if len(layers) == 1 else "layers-" + "-".join(f"{n:02d}" for n in layers)
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + "_" + tag
    return deep_update(config, update)


def update_multistream(
    config: dict,
    model_size: ModelSize,
    layers: list[int],
    depformer_order: list[int] | None = None,
    *,
    do_update_postfix: bool = True,
) -> dict:
    """Turn the run into a multi-stream SLM over ``layers`` (one codebook stream per layer)."""
    config = copy.deepcopy(config)
    clustering = config["slm"]["feature_extraction"]["clustering"]
    assert clustering == "codes", f"Multi-stream SLM requires clustering='codes', got '{clustering}'."
    n = len(layers)
    generation_order = list(depformer_order) if depformer_order is not None else None
    update = {
        "slm": {
            "feature_extraction": {
                "layers": list(layers),
                "vocab_size": [256] * n,
                "deduplicate": False,
                "cross_stream_deduplicate": True,
                "chunk_size": 4096,
            },
            "train": {
                "model": {"arch": "metasr_unit_multistream_opt", "family": "opt"},
                "model_preset": f"msopt{model_size}_4k",
                "multi_pred_loss_weight": 0.1,
                "self_distil_loss_weight": 1.0,
                "gen_loss_weight": 0.1,
                "depformer_generation_order": generation_order,
                "distill_temperature": 1.0,
                "dataset": {"max_seq_len": 4096},
            },
        }
    }
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + "_ms4k"
    return deep_update(config, update)


def update_data_parallelism(config: dict, data_parallelism: str) -> dict:
    config = copy.deepcopy(config)
    return deep_update(config, {"slm": {"train": {"trainer": {"data_parallelism": data_parallelism}}}})


def update_bf16_reduce(config: dict, *, do_update_postfix: bool = True) -> dict:
    """Reduce FSDP gradients in bf16 instead of fp32 (``trainer.fsdp.fp32_reduce=False``)."""
    config = copy.deepcopy(config)
    update = {"slm": {"train": {"trainer": {"fsdp": {"fp32_reduce": False}}}}}
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + "_bf16reduce"
    return deep_update(config, update)


def update_pack(config: dict) -> dict:
    """Enable sequence packing (``dataset.pack=True``)."""
    config = copy.deepcopy(config)
    return deep_update(config, {"slm": {"train": {"dataset": {"pack": True}}}})


def update_eval_every(config: dict, n: int) -> dict:
    """Set validate/checkpoint/publish frequency together to ``n`` steps."""
    config = copy.deepcopy(config)
    update = {
        "slm": {
            "train": {
                "regime": {
                    "validate_every_n_steps": n,
                    "checkpoint_every_n_steps": n,
                    "publish_metrics_every_n_steps": n,
                }
            }
        }
    }
    return deep_update(config, update)


def update_checkpoint_every(config: dict, n: int, *, do_update_postfix: bool = True) -> dict:
    """Checkpoint every ``n`` steps and keep every one of them.

    ``keep_checkpoint_every_n_steps`` exempts those steps from fairseq2's
    best/last pruning, so the whole series survives instead of just best+last.
    fairseq2 requires it to be a multiple of ``checkpoint_every_n_steps``.
    """
    config = copy.deepcopy(config)
    update = {
        "slm": {
            "train": {
                "regime": {
                    "checkpoint_every_n_steps": n,
                    "keep_checkpoint_every_n_steps": n,
                }
            }
        }
    }
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + f"_ckpt{n}"
    return deep_update(config, update)


def update_keep_checkpoint_every(config: dict, n: int) -> dict:
    """Retain the ``n``-step checkpoint series, independently of the save interval."""
    config = copy.deepcopy(config)
    return deep_update(config, {"slm": {"train": {"regime": {"keep_checkpoint_every_n_steps": n}}}})


_COSINE_ONLY_KEYS = [
    "cycle_len",
    "num_warmup_steps",
    "cycle_mul",
    "lr_mul",
    "start_lr",
    "final_lr",
]


def update_tristage_cosine(
    config: dict,
    stage_ratio: list[float] | None = None,
    *,
    do_update_postfix: bool = True,
) -> dict:
    """Switch the LR schedule to ``tri_stage_cosine`` (warmup -> hold -> cosine decay)."""
    config = copy.deepcopy(config)
    ratio = list(stage_ratio) if stage_ratio is not None else [0.1, 0.5, 0.4]
    if not math.isclose(sum(ratio), 1.0):
        raise ValueError(f"tri_stage_cosine stage_ratio must sum to 1.0, got {ratio} (sum {sum(ratio)}).")
    update = {
        "slm": {
            "train": {
                "lr_scheduler": {
                    "name": "tri_stage_cosine",
                    "config": {
                        "_del_": list(_COSINE_ONLY_KEYS),
                        "stage_ratio": ratio,
                        "start_lr_scale": 0.01,
                        "final_lr_scale": 0.01,
                    },
                }
            }
        }
    }
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + "_tristage"
    return deep_update(config, update)


def update_profile(config: dict, skip: int, active: int, *, do_update_postfix: bool = True) -> dict:
    """Enable fairseq2's torch profiler for a short profiling run."""
    config = copy.deepcopy(config)
    update = {
        "slm": {
            "train": {
                "common": {
                    "profilers": {
                        "torch": {
                            "enabled": True,
                            "skip_n_steps": skip,
                            "num_warmup_steps": 1,
                            "num_active_steps": active,
                            "repeat": 1,
                        }
                    }
                }
            }
        }
    }
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + "_profile"
    return deep_update(config, update)


def update_max_num_tokens(config: dict, max_num_tokens: int, *, do_update_postfix: bool = True) -> dict:
    config = copy.deepcopy(config)
    update = {"slm": {"train": {"dataset": {"max_num_tokens": max_num_tokens}}}}
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + f"_mnt{max_num_tokens}"
    return deep_update(config, update)


def update_example_shuffle_window(config: dict, window: int) -> dict:
    config = copy.deepcopy(config)
    return deep_update(config, {"slm": {"train": {"dataset": {"example_shuffle_window": window}}}})


def update_strict_data_state(config: dict) -> dict:
    """Persist the data pipeline's in-flight buffers so a requeue resumes the stream."""
    config = copy.deepcopy(config)
    return deep_update(config, {"slm": {"train": {"dataset": {"strict_data_state": True}}}})


def update_lr(config: dict, lr: float, *, do_update_postfix: bool = True) -> dict:
    config = copy.deepcopy(config)
    update = {"slm": {"train": {"optimizer": {"config": {"lr": lr}}}}}
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + f"_lr{lr:g}"
    return deep_update(config, update)


def update_num_steps(config: dict, num_steps: int, *, do_update_postfix: bool = True) -> dict:
    config = copy.deepcopy(config)
    update = {"slm": {"train": {"regime": {"num_steps": num_steps}}}}
    if do_update_postfix:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + f"_steps{num_steps}"
    return deep_update(config, update)


def update_compile(config: dict, mode: str = "default", *, do_update_postfix: bool = True) -> dict:
    config = copy.deepcopy(config)
    update = {"slm": {"train": {"model": {"compile": True, "compile_options": {"mode": mode}}}}}
    if do_update_postfix and mode != "default":
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + f"_compile-{mode}"
    return deep_update(config, update)


def update_keep_model(config: dict) -> dict:
    config = copy.deepcopy(config)
    # Retain the trained model + features after eval (otherwise the run folder is deleted).
    update = {"slm": {"eval": {"delete_data_model_after_eval": False}}}
    return deep_update(config, update)


def update_model(
    config: dict,
    checkpoint: str | None,
    model_type: ModelType,
    clustering: Clustering,
    postfix_update: str | None = None,
) -> dict:
    config = copy.deepcopy(config)
    update = {
        "slm": {"feature_extraction": {"checkpoint": checkpoint, "model_type": model_type, "clustering": clustering}}
    }
    if postfix_update is not None:
        update["slm"]["run_postfix"] = config["slm"].get("run_postfix", "") + "_" + postfix_update
    return deep_update(config, update)


def main(argv: list[str] | None = None) -> None:  # noqa: C901, PLR0912
    parser = argparse.ArgumentParser("Launch batched SLM evaluations")
    parser.add_argument("--dump", type=Path, default=os.getenv("METASR_SUBMITIT_FOLDER"), help="Submitit dump")
    parser.add_argument("--cluster", type=str, choices=get_args(ClusterName), default="FAIR_SC")
    parser.add_argument("--config_path", type=str, help="Path to the yaml configuration file", default=None)
    parser.add_argument(
        "--common_folder", type=str, help="Folder with write access to save extracted features and trained model"
    )
    parser.add_argument("--output_results_file", type=str, help="path to the output jsonl file")
    parser.add_argument("-A", "--account", type=str, default="devai", help="SLURM account")
    parser.add_argument("-Q", "--qos", type=str, default="h100_lowest", help="SLURM qos")
    parser.add_argument(
        "--data_subset",
        type=str,
        choices=sorted(DATA_SUBSETS),
        nargs="+",
        default=["6kh"],
        help="Training data subset(s)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=get_args(ModelSize),
        nargs="+",
        default=["125m"],
        help="OPT model size(s)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[6],
        help="SSL layer(s) for feature extraction."
        "Multiple layers require --multistream (one codebook stream per layer).",
    )
    parser.add_argument(
        "--depformer_order",
        type=int,
        nargs="+",
        default=None,
        help="Multi-stream only: depformer generation order.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=get_args(ModelType),
        default="spidr",
        help="Model type for feature extraction",
    )
    parser.add_argument(
        "--clustering",
        type=str,
        choices=get_args(Clustering),
        default="kmeans",
        help="Clustering method",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit path to a feature-extraction checkpoint.",
    )
    parser.add_argument(
        "--ssl_name",
        type=str,
        default=None,
        help=f"Named SSL feature extractor / checkpoint registry key (e.g. 'spidr_v2').Known: {list(MODEL_CKPTS)}.",
    )
    parser.add_argument(
        "--data_parallelism",
        type=str,
        choices=["ddp", "fsdp"],
        default="ddp",
        help="Data parallelism strategy for SLM training (default: ddp)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Override the number of nodes (default: per-model-size value).",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=None,
        help="Override per-GPU max_num_tokens (batch size).",
    )
    parser.add_argument(
        "--example_shuffle_window",
        type=int,
        default=None,
        help="Override the example shuffle buffer size, in examples. 0 shuffles the "
        "entire dataset in memory, which is infeasible at multi-million-hour scale.",
    )
    parser.add_argument(
        "--strict_data_state",
        action="store_true",
        help="Persist the data pipeline's in-flight buffers in the checkpoint so a requeued run "
        "resumes its input stream instead of restarting it.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate (default: per-model-size value).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Override training num_steps.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the SLM (unit-LM) training model.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (compile_options.mode). Use with --compile. 'max-autotune' "
        "autotunes GEMM kernels (slower compile). Tags run '_compile-<mode>'.",
    )
    parser.add_argument(
        "--bf16_reduce",
        action="store_true",
        help="Reduce FSDP gradients in bf16 instead of fp32 (trainer.fsdp.fp32_reduce=False). "
        "Halves cross-rank gradient-reduce volume; FSDP-only (DDP ignores it). Tags the run "
        "with a '_bf16reduce' postfix.",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Enable sequence packing: fixed-length packed rows + block-diagonal document "
        "masking (constant shapes for torch.compile, no padding). Supported for single- and "
        "multi-stream.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="Override validate/checkpoint/publish frequency (all three) to this many steps. "
        "Base configs default to 100.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Pin the run folder name instead of deriving it from $SLURM_JOB_ID, so a relaunch "
        "resumes from the existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=None,
        help="Checkpoint every N steps and KEEP every one of them (sets both "
        "checkpoint_every_n_steps and keep_checkpoint_every_n_steps). Without this, only "
        "the best and last checkpoints survive pruning.",
    )
    parser.add_argument(
        "--keep_checkpoint_every",
        type=int,
        default=None,
        help="Retain the N-step checkpoint series independently of --checkpoint_every; must be a multiple of it.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine_annealing",
        choices=["cosine_annealing", "tristage_cosine"],
        help="LR schedule. 'cosine_annealing' (default) is the base config's: 1k warmup then "
        "cosine decay to 0.1*peak. 'tristage_cosine' is warmup -> hold-at-peak -> cosine decay "
        "to 0.01*peak, split [0.1, 0.5, 0.4] of num_steps. Tags the run '_tristage'.",
    )
    parser.add_argument(
        "--tristage_ratio",
        type=float,
        nargs=3,
        default=None,
        metavar=("WARMUP", "HOLD", "DECAY"),
        help="Override the tri_stage_cosine warmup/hold/decay split (must sum to 1). "
        "Default: [0.1, 0.5, 0.4]. Only valid with --scheduler tristage_cosine.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable fairseq2's torch profiler (+ a CUDA memory snapshot) for a short profiling "
        "run. Use with a small --num_steps (e.g. 150). Tags the run with a '_profile' postfix.",
    )
    parser.add_argument(
        "--profile_skip",
        type=int,
        default=100,
        help="Profiler: steps to skip before profiling (clears warmup; default 100).",
    )
    parser.add_argument(
        "--profile_active",
        type=int,
        default=12,
        help="Profiler: number of steps to profile after warmup (default 12).",
    )
    parser.add_argument(
        "--keep_model",
        action="store_true",
        help="Retain the trained model + features after eval (sets delete_data_model_after_eval=false).",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Reuse pre-extracted tokens from this run folder (skips Step 1 feature extraction.",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help="SLURM wall-time limit in minutes (overrides the per-cluster default of 4320). "
        "Use a large value for long runs, e.g. 10080 for 7 days.",
    )
    parser.add_argument(
        "--multistream",
        action="store_true",
        help="Launch a multi-stream SLM",
    )
    args = parser.parse_args(argv)

    if args.tristage_ratio is not None and args.scheduler != "tristage_cosine":
        raise ValueError("--tristage_ratio is only valid with --scheduler tristage_cosine.")

    # fairseq2 needs checkpoint_every_n_steps to be a multiple of --eval_every, but
    # only raises after model init, so check it here.
    if (
        args.checkpoint_every is not None
        and args.eval_every is not None
        and args.checkpoint_every % args.eval_every != 0
    ):
        raise ValueError(
            f"--checkpoint_every ({args.checkpoint_every}) must be a multiple of --eval_every "
            f"({args.eval_every}); fairseq2 requires checkpoint_every_n_steps to be a multiple "
            "of validate/publish_metrics_every_n_steps."
        )

    if (
        args.keep_checkpoint_every is not None
        and args.checkpoint_every is not None
        and args.keep_checkpoint_every % args.checkpoint_every != 0
    ):
        raise ValueError(
            f"--keep_checkpoint_every ({args.keep_checkpoint_every}) must be a multiple of "
            f"--checkpoint_every ({args.checkpoint_every}); fairseq2 requires "
            "keep_checkpoint_every_n_steps to be a multiple of checkpoint_every_n_steps."
        )

    feature_dir = Path(args.feature_dir) if args.feature_dir else None

    def lazy_slm(
        config_path: Path,
        common_folder: Path,
        output_results_file: Path,
        config_updates: dict,
        extracted_feature_folder: Path | None = None,
    ) -> None:
        # Per-rank inductor cache: /tmp is job-private, so all ranks miss together and
        # serialise on one compile lock until inductor's 600 s timeout kills them.
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}_rank{os.environ.get('SLURM_PROCID', '0')}",
        )

        from gslm2.eval.slm.slm_pipeline import get_config_run_pipeline

        # Resolve relative to the (possibly copied) repo root: compute nodes run
        # with cwd=<copy>/src, so a repo-relative path would misresolve there.
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parents[2] / config_path

        get_config_run_pipeline(
            config_path=config_path,
            common_folder=common_folder,
            output_results_file=output_results_file,
            config_updates=config_updates,
            extracted_feature_folder=extracted_feature_folder,
            slurm_job_name=args.run_name,
        )

    if args.dump is None:
        raise ValueError("Dump directory must be specified with --dump or by setting $METASR_SUBMITIT_FOLDER")

    config_path = Path(BASE_CONFIG_PATH) if args.config_path is None else Path(args.config_path)
    common_folder = Path(args.common_folder)
    output_results_file = Path(args.output_results_file)

    if args.output_results_file:
        output_path = Path(args.output_results_file)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a"):
                pass
        except Exception as e:
            raise PermissionError(f"No write access to output_results_file: {output_path}") from e

    jobs_by_nodes: dict[int, list] = {}

    ssl_name = args.ssl_name or args.model_type
    model_type = SSL_NAME_TO_MODEL_TYPE.get(ssl_name, args.model_type)
    # The text ladder trains straight off published BPE ids, so there is no SSL
    # feature extractor to name and no layer to select.
    is_text = model_type == "text"

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    elif is_text:
        checkpoint = None
    elif ssl_name in MODEL_CKPTS:
        checkpoint = MODEL_CKPTS[ssl_name]
    else:
        raise ValueError(
            f"No checkpoint registered for '{ssl_name}'. Known: {list(MODEL_CKPTS)}. Pass --checkpoint explicitly."
        )

    layers = args.layer
    if is_text:
        if args.multistream:
            raise ValueError("--multistream is not supported for the text ladder; text is single-stream.")
        layers = []
    if len(layers) != len(set(layers)):
        raise ValueError(f"--layer must not contain duplicates, got {layers}.")
    if len(layers) > 1 and not args.multistream:
        raise ValueError(
            f"Multiple layers ({layers}) are only supported with --multistream, which predicts one "
            "codebook stream per layer. For a single-stream run pass exactly one --layer."
        )
    if args.multistream and len(layers) == 1:
        raise ValueError(
            f"--multistream needs at least two layers, got --layer {layers}. Pass the full stream "
            "set explicitly, e.g. --layer 5 6 7 8 9 10."
        )
    depformer_order = args.depformer_order
    if depformer_order is not None:
        if not args.multistream:
            raise ValueError("--depformer_order is only valid with --multistream.")
        if sorted(depformer_order) != list(range(len(layers))):
            raise ValueError(
                f"--depformer_order must be a permutation of the stream indices 0..{len(layers) - 1} "
                f"(one per --layer entry), got {depformer_order}."
            )

    if feature_dir is not None:
        num_combinations = len(args.data_subset) * len(args.model_size)
        if num_combinations != 1:
            raise ValueError(
                "--feature_dir can only be used with a single data_subset/model_size combination "
                f"(got {num_combinations}). Launch each combination separately when reusing features."
            )
        if not feature_dir.is_dir():
            raise FileNotFoundError(f"--feature_dir does not exist: {feature_dir}")

    if args.multistream and args.clustering != "codes":
        raise ValueError(
            "--multistream requires --clustering codes (multi-stream SLM predicts SSL codebook "
            f"streams, which only exist for 'codes'). Got --clustering '{args.clustering}'. "
            "Re-run with --clustering codes."
        )

    for data_subset in args.data_subset:
        for model_size in args.model_size:
            nodes = args.nodes if args.nodes is not None else get_model_size_nodes(model_size)
            d = {"slm": {}}
            d = update_model(
                d,
                checkpoint=checkpoint,
                model_type=model_type,
                clustering=args.clustering,
                postfix_update="text" if is_text else f"{ssl_name}-{args.clustering}",
            )
            d = update_data_subset(d, data_subset)
            d = update_model_size(d, model_size)
            if not is_text:
                d = (
                    update_multistream(d, model_size, layers, depformer_order)
                    if args.multistream
                    else update_layer(d, layers)
                )
            d = update_data_parallelism(d, args.data_parallelism)
            if args.max_num_tokens is not None:
                d = update_max_num_tokens(d, args.max_num_tokens)
            if args.example_shuffle_window is not None:
                d = update_example_shuffle_window(d, args.example_shuffle_window)
            if args.strict_data_state:
                d = update_strict_data_state(d)
            if args.num_steps is not None:
                d = update_num_steps(d, args.num_steps)
            if args.compile:
                d = update_compile(d, args.compile_mode)
            if args.bf16_reduce:
                d = update_bf16_reduce(d)
            if args.pack:
                d = update_pack(d)
            if args.scheduler == "tristage_cosine":
                d = update_tristage_cosine(d, args.tristage_ratio)
            if args.eval_every is not None:
                d = update_eval_every(d, args.eval_every)
            if args.checkpoint_every is not None:
                d = update_checkpoint_every(d, args.checkpoint_every)
            if args.keep_checkpoint_every is not None:
                d = update_keep_checkpoint_every(d, args.keep_checkpoint_every)
            if args.profile:
                d = update_profile(d, args.profile_skip, args.profile_active)
            if args.keep_model:
                d = update_keep_model(d)
            if args.lr is not None:
                d = update_lr(d, args.lr)
            if args.nodes is not None:
                d["slm"]["run_postfix"] = d["slm"].get("run_postfix", "") + f"_n{nodes}"
            logger.info(d)
            if nodes not in jobs_by_nodes:
                jobs_by_nodes[nodes] = []
            jobs_by_nodes[nodes].append((lazy_slm, (config_path, common_folder, output_results_file, d, feature_dir)))

    for nodes, jobs in jobs_by_nodes.items():
        logger.info("Launching %d jobs with %d node(s)", len(jobs), nodes)
        slurm_cfg = {
            "account": args.account,
            "qos": args.qos,
            "nodes": nodes,
            "gpus_per_node": 8,
        } | default_train_job_config(args.cluster)
        if args.time is not None:
            slurm_cfg["time"] = args.time
        launch_with_submitit(
            "slm",
            jobs,
            args.dump,
            slurm_cfg,
            copy_code=COPY_CODE,
        )

    # Example command to run this script:
    # python ./scripts/launchers/slm.py \
    #   --common_folder "/datasets/metaspeechssl/youssefb/slm_train/tmp" \
    #   --output_results_file "/datasets/metaspeechssl/youssefb/slm_train/slm_metrics_submitit.jsonl"
    #   --data_subset 600h \
    #   --clustering codes \
    #   --layer 5 6 7
    #
    # Example to launch scaling experiments with different data_subset and model_size combinations:
    # python ./scripts/launchers/slm.py \
    #   --common_folder "/checkpoint/devai/yosukeh/slm_eval/slm_eval_test" \
    #   --output_results_file "/checkpoint/devai/yosukeh/slm_eval/slm_eval_test/slm_metrics.jsonl" \
    #   --data_subset 600h 6kh 60kh \
    #   --model_size 125m 350m 1b 3b 7b \
    #   --clustering kmeans \
    #   --layer 6


if __name__ == "__main__":
    main()
