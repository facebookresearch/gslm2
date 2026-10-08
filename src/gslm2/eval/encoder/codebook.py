import argparse
import json
import logging
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TypedDict, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gslm2.config import ModelType
from gslm2.eval.utils import extract_codes, validate_layers
from gslm2.tools import init_logger

logger = logging.getLogger()

MAX_DIFF_PHONES_CODES: int = 2


class PhonesAndCodes(TypedDict):
    phones: list[str]
    codes: list[int]


class UnitsQuality(TypedDict):
    phone_purity: float
    cluster_purity: float
    pnmi: float
    codebook_perplexity: float
    active_codewords: int


def read_codes_and_alignments(
    units_path: Path | str, alignments_path: Path | str, sep: str = " "
) -> dict[str, PhonesAndCodes]:
    units = pd.read_json(units_path, lines=True)
    alignments = pd.read_json(alignments_path, lines=True)
    units["audio"] = units["audio"].map(lambda x: str(Path(x).resolve()))
    alignments["audio"] = alignments["audio"].map(lambda x: str(Path(x).resolve()))

    data = units.merge(alignments, on="audio", how="inner", validate="one_to_one")
    data["codes"] = data["codes"].map(lambda x: list(map(int, x.split(sep))))
    data["phones"] = data["phones"].map(lambda x: x.split(sep))
    return data.set_index("audio").to_dict(orient="index")  # type: ignore[return-value]


def count_consecutive(x: list[str] | list[int]) -> list[tuple[str | int, int]]:
    if not isinstance(x, list):
        raise TypeError("Input must be a list.")
    return [(phone, len(list(group))) for phone, group in groupby(x)]


def build_count_matrix(data: dict[str, PhonesAndCodes], ratio: int) -> tuple[np.ndarray, list[str]]:
    num_phones = len(np.unique([p for v in data.values() for p in v["phones"]]))
    codebook_size = max([max(v["codes"]) for v in data.values()]) + 1
    count = np.zeros((num_phones, codebook_size), dtype=int)

    dictionary, idx = {}, 0
    for path, phones_and_codes in data.items():
        codes = np.repeat(phones_and_codes["codes"], ratio)
        phones = phones_and_codes["phones"]
        assert abs(len(phones) - len(codes)) <= MAX_DIFF_PHONES_CODES, (len(phones), len(codes), path)
        for phone, code in zip(phones, codes, strict=False):
            if phone not in dictionary:
                dictionary[phone] = idx
                idx += 1
            count[dictionary[phone], code] += 1

    most_frequent_phones = np.argsort(count.sum(axis=1))[::-1]
    phone_order = [{v: k for k, v in dictionary.items()}[idx] for idx in most_frequent_phones]
    count = count[most_frequent_phones]
    return count, phone_order


class DiscreteUnits:
    def __init__(self, feature_size: int = 20, forced_alignment_size: int = 10) -> None:
        self.feature_size = feature_size
        self.forced_alignment_size = forced_alignment_size
        self.eps = 1e-10
        self.count: np.ndarray
        self.phone_order: list[str]
        self._proba: np.ndarray

    @property
    def ratio(self) -> int:
        return self.feature_size // self.forced_alignment_size

    def phone_purity(self) -> float:
        proba = self.count / self.count.sum()
        return proba.max(axis=0).sum()

    def cluster_purity(self) -> float:
        proba = self.count / self.count.sum()
        return proba.max(axis=1).sum()

    def pnmi(self) -> float:
        proba = self.count / self.count.sum()
        px = proba.sum(axis=1, keepdims=True)
        py = proba.sum(axis=0, keepdims=True)
        mutual_info = (proba * np.log(proba / (px @ py + self.eps) + self.eps)).sum()
        entropy_x = (-px * np.log(px + self.eps)).sum()
        return mutual_info / entropy_x

    def codebook_perplexity(self) -> float:
        proba = self.count.sum(axis=0) / self.count.sum()
        return np.exp2(-np.sum(proba * np.log2(proba + self.eps)))

    def active_codewords(self) -> int:
        return int((self.count.sum(axis=0) > 0).sum())

    def proba_phone_code(self, *, only_active: bool) -> tuple[np.ndarray, list[int]]:
        assert not np.any(np.isnan(self._proba))
        codes_order, argmax = [], self._proba.argmax(axis=0)
        for phn_idx in range(len(self.count)):
            indices = np.where(argmax == phn_idx)[0]
            codes_order.extend(indices[np.argsort(self._proba[phn_idx, indices])[::-1]])
        if only_active:
            codes_order = [c for c in codes_order if self.count[:, c].sum() > 0]
        return self._proba[:, codes_order], codes_order

    def fit(self, data: dict[str, PhonesAndCodes]) -> "DiscreteUnits":
        self.count, self.phone_order = build_count_matrix(data, self.ratio)
        count_by_code = self.count.sum(axis=0, keepdims=True)
        self._proba = np.divide(
            self.count,
            count_by_code,
            out=np.zeros_like(self.count, dtype="float64"),
            where=count_by_code != 0,
        )
        return self

    def describe(self) -> UnitsQuality:
        return {
            "phone_purity": self.phone_purity(),
            "cluster_purity": self.cluster_purity(),
            "pnmi": self.pnmi(),
            "codebook_perplexity": self.codebook_perplexity(),
            "active_codewords": self.active_codewords(),
        }

    def transcribe(self, codes: list[int]) -> str:
        phn_idx = np.argmax(self._proba, axis=0)
        return " ".join(np.repeat([self.phone_order[phn_idx[c]] for c in codes], self.ratio))

    def duration_histogram(self, data: dict[str, PhonesAndCodes]) -> tuple[pd.DataFrame, pd.DataFrame]:
        phn_idx = np.argmax(self._proba, axis=0)
        codes, phones, transcript = [], [], []

        for phones_and_codes in data.values():
            phones += count_consecutive(phones_and_codes["phones"])
            transcript += count_consecutive(self.transcribe(phones_and_codes["codes"]).split())
            codes += count_consecutive(phones_and_codes["codes"])

        phones = pd.DataFrame(phones, columns=["phone", "duration"])
        phones["duration"] *= self.forced_alignment_size
        phones["type"] = "Forced alignment"

        transcript = pd.DataFrame(transcript, columns=["phone", "duration"])
        transcript["duration"] *= self.forced_alignment_size
        transcript["type"] = "Transcription from codes"

        codes = pd.DataFrame(codes, columns=["code", "duration"])
        codes["duration"] *= self.feature_size
        codes["phone"] = [self.phone_order[phn_idx[c]] for c in codes["code"].to_numpy()]
        codes["type"] = "Discrete units"

        return pd.concat([phones, transcript]), codes


@dataclass
class CodebookConfig:
    checkpoint: Path
    model_type: ModelType
    manifest: Path
    alignments: Path
    output_dir: Path
    feature_size: int = 20  # in milliseconds
    forced_alignment_size: int = 10  # in milliseconds

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint).resolve()
        self.manifest = Path(self.manifest).resolve()
        self.alignments = Path(self.alignments).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        if self.feature_size < self.forced_alignment_size or self.feature_size % self.forced_alignment_size != 0:
            raise ValueError("Cannot handle those sizes for now.")

    @property
    def layers(self) -> list[int]:
        return validate_layers(self.checkpoint, None, codebook_proba=True)

    @property
    def name(self) -> str:
        return self.manifest.stem


def codebook_quality(cfg: CodebookConfig) -> None:
    init_logger()
    logger.info("Extract codes")
    extract_codes(cfg.checkpoint, cfg.model_type, cfg.output_dir, [cfg.manifest], cfg.layers)

    for layer in cfg.layers:
        logger.info("Units quality for layer %s", layer)
        data = read_codes_and_alignments(cfg.output_dir / f"{layer}/{cfg.name}.jsonl", cfg.alignments)
        discrete_units = DiscreteUnits(cfg.feature_size, cfg.forced_alignment_size).fit(data)
        with (cfg.output_dir / f"{layer}/{cfg.name}.json").open("w") as f:
            json.dump(discrete_units.describe(), f, indent=2)

        logger.info("Proba phone code for layer %s", layer)
        proba, _ = discrete_units.proba_phone_code(only_active=True)
        plt.figure(figsize=(8, 8))
        sns.heatmap(proba, cmap="Blues", yticklabels=discrete_units.phone_order, xticklabels=False)
        plt.savefig(cfg.output_dir / f"{layer}/{cfg.name}_proba.pdf")
        plt.close()

        logger.info("Histogram of durations for layer %s", layer)
        phone_durations, code_durations = discrete_units.duration_histogram(data)
        phone_durations = phone_durations[phone_durations["phone"] != "SIL"]
        code_durations = code_durations[code_durations["phone"] != "SIL"]

        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(8, 8), nrows=2)

        outlier = phone_durations["duration"].quantile(0.999)
        sns.histplot(
            phone_durations[(phone_durations["duration"] <= outlier)],
            x="duration",
            kde=True,
            stat="percent",
            hue="type",
            binwidth=20,
            kde_kws={"bw_adjust": 4, "cut": 0},
            ax=ax[0],
        )
        bins_left_edge = [int(rect.get_x()) for rect in ax[0].patches] + [
            int(ax[0].patches[-1].get_x() + ax[0].patches[-1].get_width())
        ]
        sns.move_legend(ax[0], "upper right", title="")
        ax[0].set_xticks(bins_left_edge)
        ax[0].set_xlabel("Duration (ms)")

        outlier = code_durations["duration"].quantile(0.999)
        sns.histplot(
            code_durations[(code_durations["duration"] <= outlier)],
            x="duration",
            stat="percent",
            hue="type",
            palette=[sns.color_palette()[2]],
            binwidth=20,
            ax=ax[1],
        )
        bins_left_edge = [int(rect.get_x()) for rect in ax[1].patches] + [
            int(ax[1].patches[-1].get_x() + ax[1].patches[-1].get_width())
        ]
        sns.move_legend(ax[1], "upper right", title="")
        ax[1].set_xticks(bins_left_edge)
        ax[1].set_xlabel("Duration (ms)")
        fig.suptitle("Phoneme duration")
        plt.savefig(cfg.output_dir / f"{layer}/{cfg.name}_duration.pdf")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebook quality")
    parser.add_argument("checkpoint", type=Path, help="Path to the checkpoint")
    parser.add_argument("model_type", type=str, choices=get_args(ModelType), help="Model type.")
    parser.add_argument("manifest", type=Path, help="Path to the manifest")
    parser.add_argument("alignments", type=Path, help="Path to the alignments")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory")
    args = parser.parse_args()

    cfg = CodebookConfig(args.checkpoint, args.model_type, args.manifest, args.alignments, args.output_dir)
    codebook_quality(cfg)


if __name__ == "__main__":
    main()
