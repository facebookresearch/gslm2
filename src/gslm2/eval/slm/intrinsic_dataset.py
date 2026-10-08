"""Dataset for loading waveform pairs for intrinsic LM evaluation.

This module provides a dataset that loads audio waveforms directly from
ZR2021 gold files or StoryCloze manifests, without requiring pre-extracted discrete units.
"""

import logging
from pathlib import Path

import pandas as pd
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ZR2021 dataset paths
SLM21_DEFAULT = Path("/datasets/metaspeechssl/shared/data/zr-data/datasets/sLM21-dataset")

# Metrics that use gold.csv files with (id, voice) or (id, voice, subtype) grouping
GOLD_PATHS = {
    "lexical_dev_copy": "lexical/dev/gold.csv",
    "lexical_dev_invocab": "lexical/dev/gold.csv",
    "lexical_dev_oov": "lexical/dev/gold.csv",
    "lexical_test_copy": "lexical/test/gold.csv",
    "lexical_test_invocab": "lexical/test/gold.csv",
    "lexical_test_oov": "lexical/test/gold.csv",
    "syntactic_dev": "syntactic/dev/gold.csv",
    "syntactic_test": "syntactic/test/gold.csv",
}

# Columns to group by for pairing (for gold.csv-based metrics)
GROUP_COLS = {
    "lexical_dev_copy": ["id", "voice"],
    "lexical_dev_invocab": ["id", "voice"],
    "lexical_dev_oov": ["id", "voice"],
    "lexical_test_copy": ["id", "voice"],
    "lexical_test_invocab": ["id", "voice"],
    "lexical_test_oov": ["id", "voice"],
    "syntactic_dev": ["id", "voice", "subtype"],
    "syntactic_test": ["id", "voice", "subtype"],
}

# StoryCloze metrics - pairs are identified by filename pattern
STORY_CLOZE_METRICS = ["sSC", "tSC"]

# All supported metrics
ALL_METRICS = list(GOLD_PATHS.keys()) + STORY_CLOZE_METRICS


class WaveformPairsDataset(Dataset):
    """Dataset that loads waveform pairs for positive/negative evaluation.

    Parses ZR2021 gold.csv files to extract positive/negative pairs,
    then loads the corresponding audio waveforms on-the-fly.

    Args:
        metric_name: Name of the metric (e.g., "lexical_dev_invocab")
        audio_root: Root directory containing the audio files
        slm21_root: Root directory of SLM21 dataset (contains gold.csv files)
        sample_rate: Target sample rate for audio loading
        max_pairs: Maximum number of pairs to load (None for all)
    """

    def __init__(
        self,
        metric_name: str,
        audio_root: Path,
        slm21_root: Path | None = None,
        sample_rate: int = 16000,
        max_pairs: int | None = None,
    ) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.audio_root = Path(audio_root)
        self.slm21_root = Path(slm21_root) if slm21_root else SLM21_DEFAULT
        self.sample_rate = sample_rate

        if metric_name not in GOLD_PATHS:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(GOLD_PATHS.keys())}")

        # Load and process gold file
        gold_path = self.slm21_root / GOLD_PATHS[metric_name]
        self.pairs = self._load_pairs(gold_path, GROUP_COLS[metric_name])

        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]

        logger.info("Loaded %d pairs for metric %s", len(self.pairs), metric_name)

    def _load_pairs(self, gold_path: Path, group_cols: list[str]) -> list[tuple[str, str]]:
        """Load positive/negative pairs from gold.csv file.

        Returns:
            List of (positive_filename, negative_filename) tuples
        """
        df_gold = pd.read_csv(gold_path)

        # Filter to groups with exactly 2 items
        grouped = df_gold.groupby(group_cols).filter(lambda x: len(x) == 2)  # noqa: PLR2004

        pairs = []
        for _, group in grouped.groupby(group_cols):
            positive = group[group["correct"] == 1].iloc[0]
            negative = group[group["correct"] == 0].iloc[0]
            pairs.append((positive["filename"], negative["filename"]))

        return pairs

    def _load_waveform(self, filename: str) -> tuple[Tensor, int]:
        """Load a waveform from the audio root directory.

        The filename from gold.csv is the stem (no extension).
        Audio files are expected to be in the audio_root directory.
        """
        # Try common audio extensions
        for ext in [".wav", ".flac", ".mp3"]:
            audio_path = self.audio_root / f"{filename}{ext}"
            if audio_path.exists():
                waveform, sr = torchaudio.load(audio_path)
                # Resample if necessary
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                return waveform.squeeze(0), self.sample_rate

        raise FileNotFoundError(f"Audio file not found for {filename} in {self.audio_root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int, int]:
        """Get a positive/negative waveform pair.

        Returns:
            (pos_waveform, neg_waveform, pos_length, neg_length)
            Waveforms are 1D tensors [samples].
        """
        pos_filename, neg_filename = self.pairs[idx]

        pos_waveform, _ = self._load_waveform(pos_filename)
        neg_waveform, _ = self._load_waveform(neg_filename)

        return pos_waveform, neg_waveform, pos_waveform.shape[0], neg_waveform.shape[0]


class WaveformPairsDatasetFromManifest(Dataset):
    """Dataset that loads waveform pairs using a manifest file for path mapping.

    This variant uses a manifest CSV to map filenames to full audio paths,
    which is useful when audio files are stored in a different structure
    than what the gold.csv filenames imply.

    Supports both:
    - ZR2021 metrics (lexical_*, syntactic_*): Uses gold.csv for pairing
    - StoryCloze metrics (sSC, tSC): Uses filename pattern (_correct/_incorrect) for pairing

    Args:
        metric_name: Name of the metric (e.g., "lexical_dev_invocab", "tSC")
        manifest_path: Path to manifest CSV with 'fileid' and 'path' columns
        slm21_root: Root directory of SLM21 dataset (contains gold.csv files)
        sample_rate: Target sample rate for audio loading
        max_pairs: Maximum number of pairs to load (None for all)
    """

    def __init__(
        self,
        metric_name: str,
        manifest_path: Path,
        slm21_root: Path | None = None,
        sample_rate: int = 16000,
        max_pairs: int | None = None,
    ) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.manifest_path = Path(manifest_path)
        self.slm21_root = Path(slm21_root) if slm21_root else SLM21_DEFAULT
        self.sample_rate = sample_rate

        if metric_name not in ALL_METRICS:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {ALL_METRICS}")

        # Load manifest and create filename -> path mapping
        manifest_df = pd.read_csv(self.manifest_path)
        if "fileid" in manifest_df.columns and "path" in manifest_df.columns:
            self.file_map = dict(zip(manifest_df["fileid"], manifest_df["path"], strict=False))
        elif "filename" in manifest_df.columns and "path" in manifest_df.columns:
            self.file_map = dict(zip(manifest_df["filename"], manifest_df["path"], strict=False))
        else:
            raise ValueError(
                f"Manifest must have 'fileid'/'filename' and 'path' columns. Found: {manifest_df.columns.tolist()}"
            )

        # Load pairs based on metric type
        if metric_name in STORY_CLOZE_METRICS:
            self.pairs = self._load_story_cloze_pairs(manifest_df)
        else:
            gold_path = self.slm21_root / GOLD_PATHS[metric_name]
            self.pairs = self._load_zr21_pairs(gold_path, GROUP_COLS[metric_name])

        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]

        logger.info("Loaded %d pairs for metric %s", len(self.pairs), metric_name)

    def _load_zr21_pairs(self, gold_path: Path, group_cols: list[str]) -> list[tuple[str, str]]:
        """Load positive/negative pairs from gold.csv file.

        Only includes pairs where BOTH filenames exist in the manifest file_map.
        """
        df_gold = pd.read_csv(gold_path)

        # Filter to only filenames that exist in the manifest
        df_gold = df_gold[df_gold["filename"].isin(self.file_map.keys())]
        logger.info("After filtering by manifest: %d rows (from gold.csv)", len(df_gold))

        # Group and filter to complete pairs (exactly 2 items)
        grouped = df_gold.groupby(group_cols).filter(lambda x: len(x) == 2)  # noqa: PLR2004

        pairs = []
        for _, group in grouped.groupby(group_cols):
            positive = group[group["correct"] == 1]
            negative = group[group["correct"] == 0]
            if len(positive) == 1 and len(negative) == 1:
                pairs.append((positive.iloc[0]["filename"], negative.iloc[0]["filename"]))

        return pairs

    def _load_story_cloze_pairs(self, manifest_df: pd.DataFrame) -> list[tuple[str, str]]:
        """Load StoryCloze pairs from manifest using filename pattern.

        StoryCloze filenames have format: {id}_{uuid}_{correct|incorrect}
        Pairs are matched by the UUID only (different ids share the same UUID).
        Example:
            1000_7ab5d7b2-f587-433f-b9b0-79a841cf0753_correct
            1001_7ab5d7b2-f587-433f-b9b0-79a841cf0753_incorrect
        """
        # Get fileid column
        if "fileid" in manifest_df.columns:
            fileids = manifest_df["fileid"].tolist()
        else:
            fileids = manifest_df["filename"].tolist()

        # Group by UUID (the middle part between first _ and _{correct|incorrect})
        pair_groups: dict[str, dict[str, str]] = {}
        for fileid in fileids:
            for suffix, label in (("_correct", "positive"), ("_incorrect", "negative")):
                if suffix not in fileid:
                    continue
                # Extract UUID: {id}_{uuid}_{correct|incorrect} -> uuid
                prefix = fileid.rsplit(suffix, 1)[0]
                parts = prefix.split("_", 1)
                if len(parts) == 2:  # noqa: PLR2004
                    pair_groups.setdefault(parts[1], {})[label] = fileid

        # Filter to complete pairs
        pairs = []
        for pair_id, group in pair_groups.items():
            if "positive" in group and "negative" in group:
                pairs.append((group["positive"], group["negative"]))
            else:
                logger.warning("Incomplete pair for %s: %s", pair_id, group)

        return pairs

    def _load_waveform(self, filename: str) -> tuple[Tensor, int]:
        """Load a waveform using the manifest path mapping."""
        if filename not in self.file_map:
            raise FileNotFoundError(f"Filename {filename} not found in manifest")

        audio_path = Path(self.file_map[filename])
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.squeeze(0), self.sample_rate

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int, int]:
        """Get a positive/negative waveform pair."""
        pos_filename, neg_filename = self.pairs[idx]

        pos_waveform, _ = self._load_waveform(pos_filename)
        neg_waveform, _ = self._load_waveform(neg_filename)

        return pos_waveform, neg_waveform, pos_waveform.shape[0], neg_waveform.shape[0]


def get_waveform_pairs_dataset(  # noqa: PLR0913
    metric_name: str,
    manifest_path: Path | None = None,
    audio_root: Path | None = None,
    slm21_root: Path | None = None,
    sample_rate: int = 16000,
    max_pairs: int | None = None,
) -> WaveformPairsDataset | WaveformPairsDatasetFromManifest:
    """Factory function to create the appropriate dataset for a metric.

    If manifest_path is provided, uses WaveformPairsDatasetFromManifest.
    Otherwise, uses WaveformPairsDataset with audio_root.

    For StoryCloze (sSC, tSC), manifest_path is required.

    Args:
        metric_name: Name of the metric (e.g., "lexical_dev_invocab", "tSC")
        manifest_path: Path to manifest CSV (required for StoryCloze)
        audio_root: Root directory of audio files (for ZR2021 metrics without manifest)
        slm21_root: Root of SLM21 dataset
        sample_rate: Target sample rate
        max_pairs: Maximum pairs to load

    Returns:
        Appropriate dataset instance
    """
    if metric_name in STORY_CLOZE_METRICS:
        if manifest_path is None:
            raise ValueError(f"StoryCloze metrics ({STORY_CLOZE_METRICS}) require manifest_path")
        return WaveformPairsDatasetFromManifest(
            metric_name=metric_name,
            manifest_path=manifest_path,
            slm21_root=slm21_root,
            sample_rate=sample_rate,
            max_pairs=max_pairs,
        )

    if manifest_path is not None:
        return WaveformPairsDatasetFromManifest(
            metric_name=metric_name,
            manifest_path=manifest_path,
            slm21_root=slm21_root,
            sample_rate=sample_rate,
            max_pairs=max_pairs,
        )

    if audio_root is None:
        raise ValueError("Either manifest_path or audio_root must be provided")

    return WaveformPairsDataset(
        metric_name=metric_name,
        audio_root=audio_root,
        slm21_root=slm21_root,
        sample_rate=sample_rate,
        max_pairs=max_pairs,
    )
