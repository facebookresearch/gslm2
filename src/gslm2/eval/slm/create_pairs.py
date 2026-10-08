import json
import logging
import sys
from pathlib import Path

import pandas as pd

from gslm2.eval.slm.pairing import (
    ML_BLIMP_BENCHMARKS,
    ML_TSC_BENCHMARKS,
    STORYCLOZE_BENCHMARKS,
    ZR21_BENCHMARKS,
    ZR21_GOLD_SUBPATH,
    ZR21_GROUP_COLS,
    is_storycloze_correct,
    storycloze_pair_id,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO, stream=sys.stdout
)
logger = logging.getLogger(__name__)


# Root for precomputed, on-the-fly create_slm_pair_manifests.py takes --audio-root from the CLI.
SLM21 = Path("/datasets/metaspeechssl/shared/data/zr-data/datasets/sLM21-dataset")


def _gold_path(benchmark: str) -> Path:
    return SLM21 / ZR21_GOLD_SUBPATH[benchmark] / "gold.csv"


def process_gold_data(df_gold: pd.DataFrame, cols_gb: list[str]) -> pd.DataFrame:
    """
    Regroups the rows in df_gold that have the same values for 'id' and 'voice'.
    Ensures there are exactly 2 rows for each combination.
    Identifies the positive item in each pair where 'correct' = 1.
    """
    grouped = df_gold.groupby(cols_gb).filter(lambda x: len(x) == 2)  # noqa: PLR2004
    grouped["is_positive"] = grouped["correct"] == 1

    return grouped


def output_jsonl(df: pd.DataFrame, output_path: Path, cols_gb: list[str]) -> int:
    """
    Saves a JSON Lines file with pairs
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    with output_path.open("w", encoding="utf-8") as f:
        for _, group in df.groupby(cols_gb):
            positive = group[group["is_positive"]].iloc[0]
            negative = group[~group["is_positive"]].iloc[0]
            json_line = {
                "positive": {
                    "units": positive["units"],
                    "file_name": positive["file_name"],
                },
                "negative": {
                    "units": negative["units"],
                    "file_name": negative["file_name"],
                },
            }
            f.write(json.dumps(json_line) + "\n")
            n_pairs += 1
    return n_pairs


def process_story_cloze(units_file_path: Path, output_jsonl_path: Path) -> None:
    # For StoryCloze the pairing is done using the filenames.

    logger.info(f"Processing file {units_file_path}")  # noqa: G004
    df_units = pd.read_json(units_file_path, lines=True)
    df_units = df_units.rename(columns={"audio": "file_name", "audio_repr": "units"})

    include_seed = units_file_path.stem.endswith("_expanded")
    df_units["is_positive"] = df_units["file_name"].apply(lambda x: is_storycloze_correct(Path(x).stem))
    df_units["pair_id"] = df_units["file_name"].apply(
        lambda x: storycloze_pair_id(Path(x).stem, include_seed=include_seed)
    )
    n_pairs = output_jsonl(df_units, output_jsonl_path, ["pair_id"])
    assert n_pairs == df_units.shape[0] / 2


def process_zr21(units_file_path: Path, gold_file_path: Path, output_jsonl_path: Path, cols_gb: list[str]) -> None:
    # For zr21 the pairing is done using the gold.csv file

    logger.info(f"Processing file {units_file_path}")  # noqa: G004
    df_units = pd.read_json(units_file_path, lines=True)
    df_units = df_units.rename(columns={"audio": "file_name", "audio_repr": "units"})
    df_units["file_name_stem"] = df_units["file_name"].apply(lambda x: Path(x).stem)
    df_gold = pd.read_csv(gold_file_path)

    # if df_units.shape[0] != df_gold.shape[0]:
    #    raise ValueError("The DataFrames do not have the same shape.")

    merged_df = pd.merge(left=df_gold, left_on="filename", right=df_units, right_on="file_name_stem", how="inner")  # noqa: PD015
    if merged_df.shape[0] != df_units.shape[0]:
        raise ValueError("Merged DataFrame does not have the same number of rows as the original DataFrames.")

    processed_gold = process_gold_data(merged_df, cols_gb)
    _ = output_jsonl(processed_gold, output_jsonl_path, cols_gb)
    # assert n_pairs == df_gold.shape[0] / 2


def process_multilingual_pairs(units_file_path: Path, output_jsonl_path: Path) -> None:
    logger.info(f"Processing multilingual file {units_file_path}")  # noqa: G004
    df_units = pd.read_json(units_file_path, lines=True)
    df_units = df_units.rename(columns={"audio": "file_name", "audio_repr": "units"})

    df_units["stem"] = df_units["file_name"].apply(lambda x: Path(x).stem)
    df_units["is_positive"] = df_units["stem"].apply(lambda x: x.rsplit("_", 1)[-1] == "correct")
    df_units["pair_id"] = df_units["stem"].apply(lambda x: x[x.index("_") + 1 : x.rindex("_")])
    n_pairs = output_jsonl(df_units, output_jsonl_path, ["pair_id"])
    assert n_pairs == df_units.shape[0] / 2


def create_paired_file(input_path: Path) -> None:
    filename = input_path.stem
    output_path = input_path.parent / "pairs" / input_path.name
    if filename in STORYCLOZE_BENCHMARKS:
        process_story_cloze(input_path, output_path)
    elif filename in ZR21_BENCHMARKS:
        process_zr21(
            units_file_path=input_path,
            gold_file_path=_gold_path(filename),
            output_jsonl_path=output_path,
            cols_gb=list(ZR21_GROUP_COLS[filename]),
        )
    elif filename in ML_TSC_BENCHMARKS or filename in ML_BLIMP_BENCHMARKS:
        process_multilingual_pairs(input_path, output_path)
    else:
        logger.error("Unknown test filename: %s", filename)
        raise ValueError(f"Unknown test filename: {filename}")
