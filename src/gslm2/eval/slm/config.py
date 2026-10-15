"""SLM eval configs: feature extraction, k-means, and fairseq2 train/eval configs."""

import json
import logging
import os
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

if TYPE_CHECKING:
    from gslm2.fs2.unit_lm.recipes.train.config import UnitLMTrainConfig


def _load_fs2_config_helpers():  # noqa: ANN202
    """Import fairseq2-backed config helpers lazily (fairseq2 is an optional extra)."""
    from gslm2.fs2.unit_lm.recipes.train.config import load_unit_lm_train_config
    from gslm2.fs2.unit_lm.recipes.train.dataset import LMDatasetConfig
    from gslm2.fs2.unit_lm.recipes.train.dataset_speech_manifest import ManifestUnitDatasetConfig
    from gslm2.fs2.unit_lm.recipes.train.dataset_text import TextUnitDatasetConfig
    from gslm2.fs2.unit_lm.recipes.train.storage_s3_eoz import resolve_manifest_root

    return (
        load_unit_lm_train_config,
        LMDatasetConfig,
        ManifestUnitDatasetConfig,
        TextUnitDatasetConfig,
        resolve_manifest_root,
    )


logger = logging.getLogger(__name__)

ModelType = tp.Literal["spidr", "hubert", "wav2vec2", "wavlm", "text"]


def _resolve_paths(paths: list[Path] | None) -> list[Path]:
    if paths is None:
        return []
    return [Path(path).resolve() for path in paths]


def get_slurm_job_id_or_datetime() -> str:
    job_id = os.getenv("SLURM_JOB_ID")

    if job_id is not None:
        return f"SLM_{job_id}"
    # Use UTC timezone for datetime
    return str(datetime.now(timezone.utc).strftime("%Y%m%d_%HM"))  # noqa: UP017


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
class FeatExtractConfig:
    checkpoint: Path | None
    model_type: ModelType
    manifests_train: list[Path]
    manifests_valid: list[Path]
    manifests_test: list[Path]
    output_folder: Path
    layers: list[int]
    deduplicate: bool = True  # Deduplicate each unit sequence independently.
    cross_stream_deduplicate: bool = False  # Deduplicate aligned multi-stream codes jointly.
    chunk: bool = True  # Chunk unit sequences after optional deduplication.
    chunk_size: int = 2048  # Max length per chunk when chunk=True
    overwrite_folder: bool = False
    distributed_port: int = 58555
    batch_size: int = 8
    clustering: Literal["codes", "kmeans"] = "codes"
    manifests_train_kmeans: list[Path] | None = None
    manifests_unit_vocoder: list[Path] | None = None
    # text config
    text_remove_license: bool = False
    text_sentencize: bool = True
    text_add_special_tokens: bool = True
    text_tokenizer_path: str | None = None
    vocab_size: int | list[int] = 256
    # Kmeans config
    kmeans: KMConfig | dict[str, tp.Any] | None = None
    kmeans_pooling_width: int = 1
    encoder_dtype: str = "float32"

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint).resolve() if self.checkpoint is not None else None
        self.output_folder = Path(self.output_folder).resolve()
        self.manifests_train = _resolve_paths(self.manifests_train)
        self.manifests_valid = _resolve_paths(self.manifests_valid)
        self.manifests_test = _resolve_paths(self.manifests_test)
        self.manifests_train_kmeans = _resolve_paths(self.manifests_train_kmeans)
        self.manifests_unit_vocoder = _resolve_paths(self.manifests_unit_vocoder)

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


@dataclass
class EvalConfig:
    common_folder: Path
    output_results_file: Path
    metric_names: list[str]
    metadata: dict = field(default_factory=dict)  # metadata to save in the results file
    max_pairs: int | None = None
    batch_size: int | None = 32
    seed: int = 0
    vocab_size: int = 256
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
    cfg_feat_ext: FeatExtractConfig = field(default_factory=FeatExtractConfig)
    cfg_trainer_fs2: "UnitLMTrainConfig" = field(
        default_factory=lambda: __import__(
            "gslm2.fs2.unit_lm.recipes.train.config", fromlist=["UnitLMTrainConfig"]
        ).UnitLMTrainConfig()
    )
    cfg_eval: EvalConfig = field(default_factory=EvalConfig)


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
        lm_dataset_config,
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
        cfg_trainer_fs2.dataset.config_overrides = lm_dataset_config(path=feature_dir)
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
    path = Path(path).resolve()
    with path.open() as f:
        cfgdata = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)

    if dict_update is None:
        dict_update = {}
    cfgdata = deep_update(cfgdata, dict_update)

    # global seed for the pipeline
    global_seed = cfgdata["slm"].get("seed", 0)

    if slurm_job_name is None:
        slurm_job_name = get_slurm_job_id_or_datetime()

    if cfgdata["slm"].get("run_postfix", None) is not None:
        slurm_job_name += "_" + cfgdata["slm"].get("run_postfix")

    if output_results_file is None:
        output_results_file = cfgdata["slm"]["output_results_file"]

    if common_folder is None:
        common_folder = Path(cfgdata["slm"]["common_folder"])
    common_folder = common_folder / slurm_job_name

    # Feature extraction
    cfg_feat_ext = FeatExtractConfig(**cfgdata["slm"]["feature_extraction"], output_folder=common_folder)
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
