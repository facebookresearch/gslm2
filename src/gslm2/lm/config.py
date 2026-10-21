"""Preparation, recipe, and evaluation settings for fairseq2 LM pipelines."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from gslm2.encoder.units.config import FeatExtractConfig, layer_num_str

if TYPE_CHECKING:
    from gslm2.lm.recipes.train.config import UnitLMTrainConfig


def _load_fs2_config_helpers():  # noqa: ANN202
    """Import fairseq2-backed config helpers lazily (fairseq2 is an optional extra)."""
    from gslm2.lm.data.base import LMDatasetConfig
    from gslm2.lm.data.manifest import ManifestUnitDatasetConfig
    from gslm2.lm.data.storage import resolve_manifest_root
    from gslm2.lm.data.text import TextUnitDatasetConfig
    from gslm2.lm.recipes.train.config import load_unit_lm_train_config

    return (
        load_unit_lm_train_config,
        LMDatasetConfig,
        ManifestUnitDatasetConfig,
        TextUnitDatasetConfig,
        resolve_manifest_root,
    )


logger = logging.getLogger(__name__)


def get_slurm_job_id_or_datetime() -> str:
    run_name = os.getenv("SLM_RUN_NAME")
    if run_name is not None:
        return run_name
    job_id = os.getenv("SLURM_JOB_ID")

    if job_id is not None:
        return f"SLM_{job_id}"
    # Use UTC timezone for datetime
    return str(datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))  # noqa: UP017


@dataclass
class LMPreparationConfig(FeatExtractConfig):
    """LM data preparation, including text tokenization."""

    text_remove_license: bool = False
    text_sentencize: bool = True
    text_add_special_tokens: bool = True
    text_tokenizer_path: str | None = None
    slm21_root: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.slm21_root is not None:
            self.slm21_root = Path(self.slm21_root).expanduser().resolve()


@dataclass
class EvalConfig:
    common_folder: Path
    output_results_file: Path
    metric_names: list[str]
    metadata: dict = field(default_factory=dict)  # metadata to save in the results file
    max_pairs: int | None = None
    batch_size: int | None = 32
    seed: int = 0
    vocab_size: int | list[int] = 256
    offset: int = 2
    bos: int | None = None
    local_rank: int = -1
    feature_dir_override: Path | None = None
    eval_pairs_dir: Path | None = None
    delete_data_model_after_eval: bool = (
        True  # if True: deletes the feature extraction data and the SLM model after evaluation
    )
    model_preset: str = "opt125m"
    distill_temperature: float = 1.0
    depformer_generation_order: list[int] | None = None
    model_config_overrides: dict | None = None

    def __post_init__(self) -> None:
        self.common_folder = Path(self.common_folder).resolve()
        self.output_results_file = Path(self.output_results_file).resolve()
        self.output_results_file.parent.mkdir(parents=True, exist_ok=True)
        if self.feature_dir_override is not None:
            self.feature_dir_override = Path(self.feature_dir_override).resolve()
        if self.eval_pairs_dir is not None:
            self.eval_pairs_dir = Path(self.eval_pairs_dir).resolve()


@dataclass
class SLMConfig:
    cfg_feat_ext: LMPreparationConfig
    cfg_trainer_fs2: "UnitLMTrainConfig"
    cfg_eval: EvalConfig


def deep_update(dict1: dict, dict2: dict) -> dict:
    """Recursively merge dict2 into dict1.

    Honors fairseq2's ``_del_`` convention (see ``fairseq2.utils.config``).
    """
    for key in dict2.get("_del_", []):
        dict1.pop(key, None)
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_update(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def _read_experiment(path: str | Path, dict_update: dict | None = None) -> dict:
    path = Path(path).resolve()
    with path.open() as f:
        cfgdata = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    if not isinstance(cfgdata, dict) or "slm" not in cfgdata:
        raise ValueError("End-to-end LM configs must contain an 'slm' section.")
    return deep_update(cfgdata, dict_update or {})


def _make_preparation_config(cfgdata: dict, common_folder: Path | None, slurm_job_name: str) -> LMPreparationConfig:
    root = common_folder or Path(cfgdata["slm"]["common_folder"])
    cfg = LMPreparationConfig(**cfgdata["slm"]["feature_extraction"], output_folder=root / slurm_job_name)
    cfg.cfg_kmeans.seed = cfgdata["slm"].get("seed", 0)
    return cfg


def get_preparation_config(path: str | Path) -> LMPreparationConfig:
    """Load preparation settings without importing fairseq2."""
    cfgdata = _read_experiment(path)
    job_name = get_slurm_job_id_or_datetime()
    if cfgdata["slm"].get("run_postfix") is not None:
        job_name += "_" + cfgdata["slm"]["run_postfix"]
    return _make_preparation_config(cfgdata, None, job_name)


def _apply_dataset_source(  # noqa: PLR0913
    cfg_trainer_fs2: "UnitLMTrainConfig",
    *,
    token_manifest: dict | None,
    text_tokens: dict | None,
    feature_dir: Path,
    layers: list[int],
    layer_number: str,
    is_multistream_codes: bool,
) -> Path | None:
    """Point the trainer at whichever of the three data sources the config names
    (TextUnitDataset, UnitDataset, ManifestUnitDatasetConfig).

    Returns the directory holding the eval pairs, or ``None`` when the
    pairs are tokenized into the run folder rather than read from a published tree.
    """
    (
        _,
        _,
        manifest_unit_dataset_config,
        text_unit_dataset_config,
        resolve_manifest_root,
    ) = _load_fs2_config_helpers()
    if token_manifest is not None and text_tokens is not None:
        raise ValueError("token_manifest and text_tokens are mutually exclusive; set exactly one.")

    if text_tokens is not None:
        cfg_trainer_fs2.dataset.family = "metasr_text"
        cfg_trainer_fs2.dataset.config_overrides = text_unit_dataset_config(
            path=resolve_manifest_root(Path(text_tokens["path"])),
        )
        eval_pairs_root = resolve_manifest_root(Path(text_tokens["eval_pairs_root"]))
        cfg_trainer_fs2.dataset.valid_metrics.base_dir = eval_pairs_root
        return eval_pairs_root / "test" / "pairs"

    if token_manifest is None:
        cfg_trainer_fs2.dataset.path = feature_dir
        cfg_trainer_fs2.dataset.config_overrides = {"path": str(feature_dir)}
        cfg_trainer_fs2.dataset.valid_metrics.base_dir = feature_dir
        return None

    # Pairs live under <mode>/<layer>, mirroring the tokenizer's output tree. Relative
    # roots resolve under the bbfs profile, as with token_manifest.path.
    mode = "multi_stream" if is_multistream_codes else "single_stream"
    eval_pairs_dir = resolve_manifest_root(Path(token_manifest["eval_pairs_root"])) / mode / layer_number
    if not (eval_pairs_dir / "test" / "pairs").is_dir():
        raise FileNotFoundError(
            f"No benchmark pairs at {eval_pairs_dir / 'test' / 'pairs'}. Pairs exist only "
            "for the layers they were tokenized for."
        )
    cfg_trainer_fs2.dataset.family = "metasr_manifest_unit"
    cfg_trainer_fs2.dataset.config_overrides = manifest_unit_dataset_config(
        path=resolve_manifest_root(Path(token_manifest["path"])),
        layers=layers,
        reset_data_reader=bool(token_manifest.get("reset_data_reader")),
    )
    cfg_trainer_fs2.dataset.valid_metrics.base_dir = eval_pairs_dir
    return eval_pairs_dir


def get_slm_config(  # noqa: PLR0913
    path: str | Path,
    dict_update: dict | None = None,
    common_folder: Path | None = None,
    output_results_file: Path | None = None,
    slurm_job_name: str | None = None,
    extracted_feature_folder: Path | None = None,
) -> SLMConfig:
    cfgdata = _read_experiment(path, dict_update)

    # global seed for the pipeline
    global_seed = cfgdata["slm"].get("seed", 0)

    if slurm_job_name is None:
        slurm_job_name = get_slurm_job_id_or_datetime()

    if cfgdata["slm"].get("run_postfix", None) is not None:
        slurm_job_name += "_" + cfgdata["slm"].get("run_postfix")

    if output_results_file is None:
        output_results_file = cfgdata["slm"]["output_results_file"]

    cfg_feat_ext = _make_preparation_config(cfgdata, common_folder, slurm_job_name)
    common_folder = cfg_feat_ext.output_folder
    # Steps 2 and 3 use layer_folder. Multi-stream codes are written under the
    # first layer so downstream training/eval can read one aligned stream set.
    layer_number = layer_num_str(cfg_feat_ext.effective_output_layers)  # "" if no layer
    layer_folder = common_folder / layer_number
    feature_dir = (extracted_feature_folder or common_folder) / layer_number
    cfg_feat_ext.cfg_kmeans.seed = global_seed

    # Train
    if "common" not in cfgdata["slm"]["train"]:
        cfgdata["slm"]["train"]["common"] = {}
    cfgdata["slm"]["train"]["common"]["seed"] = global_seed

    (load_unit_lm_train_config, *_) = _load_fs2_config_helpers()
    cfg_trainer_fs2 = load_unit_lm_train_config(cfgdata["slm"]["train"])
    cfg_trainer_fs2.model.vocab_size = cfg_feat_ext.vocab_size
    cfg_trainer_fs2.common.metric_recorders.wandb.run_name = slurm_job_name

    # fairseq2 treats model.path as an init for a fresh run (step 0, new optimizer),
    # not a resume. Point it at a step_N/model directory.
    init_model_path = cfgdata["slm"].get("init_model_path")
    if init_model_path:
        cfg_trainer_fs2.model.path = Path(init_model_path)
        logger.info("Finetune-from-pretrained: init LM weights from %s", init_model_path)

    # A token_manifest means the units are already extracted and published
    token_manifest = cfgdata["slm"].get("token_manifest") or None
    # text_tokens means BPE ids are already published; only the eval pairs are
    # tokenized at startup.
    text_tokens = cfgdata["slm"].get("text_tokens") or None
    eval_pairs_dir = _apply_dataset_source(
        cfg_trainer_fs2,
        token_manifest=token_manifest,
        text_tokens=text_tokens,
        feature_dir=feature_dir,
        layers=list(cfg_feat_ext.layers),
        layer_number=layer_number,
        is_multistream_codes=cfg_feat_ext.is_multistream_codes,
    )

    # Eval
    metric_names = [f.stem for f in cfg_feat_ext.manifests_test]
    cfg_eval = EvalConfig(
        **dict(cfgdata["slm"]["eval"]),
        common_folder=layer_folder,
        output_results_file=output_results_file,
        metric_names=metric_names,
        metadata={"ssl_ckpt": cfgdata["slm"]["feature_extraction"]["checkpoint"], "run_name": slurm_job_name},
        vocab_size=cfg_feat_ext.vocab_size,
        feature_dir_override=None if token_manifest else feature_dir,
        eval_pairs_dir=eval_pairs_dir,
        seed=global_seed,
        model_preset=cfg_trainer_fs2.model_preset,
        distill_temperature=cfg_trainer_fs2.distill_temperature,
        depformer_generation_order=cfg_trainer_fs2.depformer_generation_order,
        model_config_overrides=cfg_trainer_fs2.model.config_overrides,
    )
    cfg_eval.seed = global_seed

    return SLMConfig(cfg_feat_ext=cfg_feat_ext, cfg_trainer_fs2=cfg_trainer_fs2, cfg_eval=cfg_eval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path)
    cfg_slm = get_slm_config(config_path)
