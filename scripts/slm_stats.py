"""Significance tooling for GSLM2 SLM results, calibrated on the SpidR 2.0 seed study.

Two source sheets in ``results/sheets/paper-google-sheets-source/`` hold an 8-layer x 8-seed
grid each:

* ``... (seed-0) SLM (seed-N).csv`` -- encoder fixed, SLM seed varied  -> sigma_SLM
* ``... (seed-N) SLM (seed-0) (1).csv`` -- encoder varied, SLM seed fixed -> sigma_enc

Both grids measure a component *directly*; sigma_tot = sqrt(sigma_enc^2 + sigma_SLM^2) is derived
from them and is the spread of a fully independent replication (see below).

Layers are *fixed evaluation points*, not replicates: each layer keeps its own mean. Only the
noise level is assumed shared across layers, and that assumption is tested (Bartlett) rather
than asserted. The model is

    y[metric, layer, seed] = mu[metric, layer] + eps,   eps ~ N(0, sigma[metric]^2)

Residuals correlate across layers -- a bad encoder seed tends to be bad at several layers -- so
the pooled variance carries fewer than ``n_layers * (n_seeds - 1)`` degrees of freedom. The
effective count, ``df_eff = L * (n - 1) / (1 + (L - 1) * mean_offdiag(rho^2))``, is retained
internally because a confidence interval for a variance needs one, but it plays no part in the
decision rule and is not reported in the paper (see ``_z``).

Every other model in the paper has a single encoder seed, so the primary instrument is a
calibrated threshold that borrows sigma from this study:

    Delta / (sigma_hat * sqrt(1/k_A + 1/k_B))  compared against a normal quantile

Why the encoder grid measures sigma_enc and not sigma_enc + sigma_SLM
---------------------------------------------------------------------
Write the score as Y(e, s) for encoder seed e and pipeline seed s. The encoder grid is
sd_e[Y(e, 0)] and the SLM grid is sd_s[Y(0, s)]. Under Y = mu + alpha_e + beta_s the first *is*
sigma_alpha: beta_0 is a constant within that grid and contributes nothing, so there is no
sigma_SLM^2 term inside it to subtract. Correspondingly sigma_tot = sqrt(sigma_enc^2 +
sigma_SLM^2) -- addition, not subtraction -- and it answers "how far apart would two independent
replications of this recipe land", which is the quantity the thresholds below need.

The SLM-training noise that *does* vary across e (batch composition, and the dropout masks
downstream of it) is a mediator on the path from encoder seed to score, not an independent second
draw, so it belongs inside alpha_e. Audited against the four sources of Dufour et al. (2026), in
src/metasr_eval/slm and src/metasr/fs2/unit_lm:

1. Weight initialisation -- FROZEN. fairseq2's _TorchConfigurer._set_seed() calls
   rng_bag.manual_seed(common.seed) before the model is built, and vocab_size is pinned to 256 by
   the feature-extraction config, so every parameter shape and hence every initial tensor is
   identical across encoder seeds.
2. Example order -- VARIES. The file-list and example shuffles use the same seed, but
   example_shuffle_window=0 is a full-stream shuffle whose permutation depends on the stream
   length, and deduplicate=True (torch.unique_consecutive) plus fixed-size chunking make that
   length encoder-dependent. filter(check_len) and bucket_by_length(max_num_tokens=81920) then
   compose different batches: 72.5% of utterances have a different deduplicated length between two
   checkpoints 20k steps apart in a *single* run, which is a weaker perturbation than a reseed.
3. Loss-injected noise -- ABSENT as an independent source. Cross-entropy injects nothing, unlike
   the flow-matching objective of that paper, which draws a fresh eps ~ N(0, I) each step. The
   only stochastic op is OPTConfig.dropout_p = 0.1, and its masks are a deterministic function of
   (seed, RNG consumption order, tensor shapes): identical stream => bitwise identical masks. They
   decorrelate here only because (2) diverges, i.e. this class is downstream of (2), not parallel
   to it.
4. Hardware non-determinism -- SUPPRESSED IN TRAINING AND EVAL, NOT IN FEATURE EXTRACTION.
   is_train_deterministic defaults to True -> use_deterministic_algorithms(True),
   cudnn.benchmark=False, CUBLAS_WORKSPACE_CONFIG set. But that call lives in
   UnitLMTrainRecipe.create_trainer, i.e. pipeline step 2; step 1 has already run by then, with
   only CUBLAS_WORKSPACE_CONFIG (slm_pipeline.py, and it disagrees: ":4096:8" there vs ":16:8" in
   the recipe). Extraction therefore runs with cudnn.benchmark=True. Measured, not assumed: see
   sigma_repro below.

So (1) contributes nothing, (3) is not a source at all, (2) mediates and belongs to alpha_e, and
(4) contributes only via step 1. The grid is a clean estimate of sigma_enc up to sigma_repro.

The irreproducibility floor sigma_repro
---------------------------------------
Eight runs of the whole pipeline at one seed on one encoder checkpoint (see DETERMINISM_JSONL)
produced exactly two outcomes -- six runs bitwise identical, two bitwise identical -- and the md5s
of their extracted token stores partition the runs the same 6/2 way. So training and evaluation are
bitwise reproducible given identical units, and the entire residual comes from step 1: a cuDNN
autotuning difference flips a unit at a codebook near-tie, which changes a deduplicated length,
which re-buckets the batch stream and forks the run. That is why the outcomes are discrete rather
than jittery. sigma_repro is 0.02-0.15 points depending on the metric, at most 14% of sigma_enc,
and it sits inside BOTH grids, so it should be subtracted from both or from neither. Caveats: 8
runs at one layer (7 df) against the studies' layer-pooled 8x8, and the underlying variable is
bimodal, so the sd is a crude summary of "2 runs in 8 forked".

Not a fifth source for these runs: clustering defaults to "codes", which takes the units straight
from the encoder's codebook assignments (extract_features.save_units) and never calls learn_kmeans,
so cfg_kmeans.seed is dead config. Under clustering="kmeans" the tokeniser would be refit per run
and would need separate treatment.

Usage::

    python scripts/slm_stats.py decompose            # noise floor -> results/sheets/slm_noise_floor.csv
    python scripts/slm_stats.py power                # minimum detectable difference vs #seeds
    python scripts/slm_stats.py compare --metric swuggy_test_all --a 71.2 --b 69.8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO / "results" / "sheets" / "paper-google-sheets-source"
SLM_SEED_CSV = SOURCE_DIR / "GSLM2 SLM results - SpidR 2.0 (seed-0) SLM (seed-N).csv"
ENC_SEED_CSV = SOURCE_DIR / "GSLM2 SLM results - SpidR 2.0 (seed-N) SLM (seed-0) (1).csv"
NOISE_FLOOR_CSV = REPO / "results" / "sheets" / "slm_noise_floor.csv"
SIGMA_RECIPE_CSV = REPO / "results" / "sheets" / "slm_sigma_by_recipe.csv"
BEAT_CSV = REPO / "results" / "sheets" / "slm_gap_to_beat.csv"

# Determinism control: 8 runs of the whole pipeline at ONE seed on ONE encoder checkpoint
# (spidr-2.0/seed-0__final.pt, layer 7, codes). Measures the irreproducibility floor sigma_repro.
DETERMINISM_JSONL = REPO / "results" / "slm" / "slm__determinism-seed0.jsonl"

# jsonl accuracy field -> metric key. Composites are rebuilt below from these, matching the source
# sheets: swuggy_*_all = mean(in-vocab, OOV) and test_avg = mean(sWUGGY all, sBLIMP test, tSC test),
# both verified against the seed-study grids to within their two-decimal rounding.
DETERMINISM_FIELDS = {
    "lexical_dev_invocab_accuracy": "swuggy_dev_invocab",
    "lexical_dev_oov_accuracy": "swuggy_dev_oov",
    "lexical_test_invocab_accuracy": "swuggy_test_invocab",
    "lexical_test_oov_accuracy": "swuggy_test_oov",
    "syntactic_dev_accuracy": "sblimp_dev",
    "syntactic_test_accuracy": "sblimp_test",
    "sSC_accuracy": "ssc",
    "tSC_accuracy": "tsc_test",
    "tSC_dev_expanded_accuracy": "tsc_dev_expanded",
}

# Further 8-encoder-seed studies, used to check whether sigma_enc transfers between recipes.
# These sheets have one row per (experiment, seed, layer) with metrics as fractions, and a single
# SLM run per encoder seed -- so, like the main encoder grid, they measure sigma_enc.
# ``normalised`` records whether the recipe includes the encoder-output normalisation step.
RECIPE_SHEETS = (
    REPO / "results" / "sheets" / "ablations" / "slm.csv",
    REPO / "results" / "sheets" / "dinosr-reproduced" / "slm.csv",
    REPO / "results" / "sheets" / "paper-google-sheets-source" / "Detailed encoder results - SpidR reproduced.csv",
)
RECIPES: dict[str, tuple[str, bool]] = {
    "6kh__full__dinosr-codes__default": ("DinoSR (reproduced)", False),
    "full__spidr__default": ("SpidR (reproduced)", False),
    "6kh__full__spidr__+shared-heads": (r"\quad $+$ shared heads", False),
    "6kh__full__spidr__+shared-heads+reg": (r"\quad $+$ regularisation", False),
    "6kh__full__spidr__+shared-heads+reg+norm": (r"\quad $+$ normalisation", True),
}
# Column in the recipe sheets -> metric key in METRICS. The recipe sheets carry a single tSC
# column, which matches the test-set tSC of the main study.
RECIPE_COLUMNS = {
    "sWUGGY_avg_dev": "swuggy_dev_all",
    "sWUGGY_avg_test": "swuggy_test_all",
    "sBLIMP_dev": "sblimp_dev",
    "sBLIMP_test": "sblimp_test",
    "tSC": "tsc_test",
    "sSC": "ssc",
}
# Metrics shown in the cross-recipe table, in order.
RECIPE_TABLE_METRICS = ("swuggy_test_all", "sblimp_test", "tsc_test", "ssc", "test_avg")

# Column index in the source sheets -> (key, display name, number of benchmark pairs).
# Pair counts are the row counts of the built pair files in
# /datasets/metaspeechssl/shared/assets/manifests/sLM21-dataset/pairs/.
# The two averages are composites of the columns above them and have no single pair count.
METRICS: dict[int, tuple[str, str, int | None]] = {
    2: ("swuggy_dev_invocab", "sWuggy dev in-vocab", 20_000),
    3: ("swuggy_dev_oov", "sWuggy dev OOV", 20_000),
    4: ("swuggy_dev_all", "sWuggy dev all", 40_000),
    5: ("swuggy_test_invocab", "sWuggy test in-vocab", 80_000),
    6: ("swuggy_test_oov", "sWuggy test OOV", 80_000),
    7: ("swuggy_test_all", "sWuggy test all", 160_000),
    8: ("sblimp_dev", "sBLIMP dev", 25_200),
    9: ("sblimp_test", "sBLIMP test", 63_000),
    10: ("ssc", "sSC", 1_871),
    11: ("tsc_dev", "tSC dev", 1_871),
    12: ("tsc_dev_expanded", "tSC dev+", 18_710),
    13: ("tsc_test", "tSC test", 1_871),
    14: ("dev_avg", "dev avg.", None),
    15: ("test_avg", "test avg.", None),
}

# Headline endpoints. Test tSC is kept over the lower-noise expanded development set because it is
# what the prior literature reports, so dropping it would break comparability with models whose
# checkpoints are not released.
PRIMARY_METRICS = ("swuggy_test_all", "sblimp_test", "tsc_test")

DATA_ROW_PREFIX = "spidr2-slm_codes-l"
ALPHA = 0.05
POWER = 0.80
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0
MAX_EXACT_PERMUTATIONS = 200_000


# --------------------------------------------------------------------------------------- parsing


def _z(q: float) -> float:
    """Standard-normal quantile.

    The decision rule uses normal rather than t quantiles. sigma is estimated from 64 runs per
    grid (and from six such grids when pooled across recipes), so the t correction is worth
    4-7% on the thresholds; using the normal quantile keeps the rule stateless -- no degrees of
    freedom to carry around or justify -- at a cost small relative to how much sigma itself
    varies between recipes.
    """
    return float(stats.norm.ppf(q))


def _parse_percent(text: str) -> float | None:
    text = text.strip()
    if not text.endswith("%"):
        return None
    try:
        return float(text[:-1])
    except ValueError:
        return None


def load_seed_study(path: Path) -> dict[str, np.ndarray]:
    """Read one source sheet into ``{metric_key: array of shape (n_layers, n_seeds)}``.

    Rows 0-2 are merged headers. Section rows name the varied seed (``SLM (seed-3)`` or
    ``SpidR 2.0 (seed-3)``); the trailing ``Mean`` / ``Std`` / ``Median`` / ``[Q1, Q3]``
    sections are summaries computed by the spreadsheet and are skipped. Layers are recovered
    from the run name and sorted, so the row order in the sheet does not matter.
    """
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    # {metric_key: {layer: {seed: value}}}
    collected: dict[str, dict[int, dict[int, float]]] = {key: {} for key, _, _ in METRICS.values()}
    seed: int | None = None

    for row in rows[3:]:
        if not row or not row[0]:
            continue
        name = row[0]
        if not name.startswith(DATA_ROW_PREFIX):
            # Section header: a seed block, or one of the spreadsheet summary blocks.
            seed = int(name.rsplit("seed-", 1)[1].rstrip(")")) if "seed-" in name else None
            continue
        if seed is None:
            continue
        layer = int(name[len(DATA_ROW_PREFIX) :].split("-", 1)[0])
        for index, (key, _, _) in METRICS.items():
            value = _parse_percent(row[index]) if index < len(row) else None
            if value is not None:
                collected[key].setdefault(layer, {})[seed] = value

    study: dict[str, np.ndarray] = {}
    for key, by_layer in collected.items():
        if not by_layer:
            continue
        layers = sorted(by_layer)
        seeds = sorted(by_layer[layers[0]])
        if any(sorted(by_layer[layer]) != seeds for layer in layers):
            raise ValueError(f"{path.name}: ragged layer/seed grid for metric {key!r}")
        study[key] = np.array([[by_layer[layer][s] for s in seeds] for layer in layers], dtype=float)
    return study


def load_recipe_studies(paths: tuple[Path, ...] = RECIPE_SHEETS) -> dict[str, dict[str, np.ndarray]]:
    """Read the per-recipe seed sheets into ``{recipe: {metric: (n_layers, n_seeds)}}``.

    Rows whose ``seed`` is not an integer are the sheets' own min/mean/median/max/std summaries
    and are skipped, as are rows with no SLM metrics (the SpidR sheet also covers layers 1-4,
    where only encoder metrics were computed). Recipes with fewer than two seeds are dropped:
    a single-seed recipe carries no variance information.
    """
    cells: dict[str, dict[str, dict[tuple[int, int], float]]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["seed"].isdigit() or not row.get("sWUGGY_avg_test"):
                    continue
                key = (int(row["layer"]), int(row["seed"]))
                by_metric = cells.setdefault(row["experiment"], {})
                for column, metric in RECIPE_COLUMNS.items():
                    by_metric.setdefault(metric, {})[key] = float(row[column]) * 100.0

    studies: dict[str, dict[str, np.ndarray]] = {}
    for recipe, by_metric in cells.items():
        grids: dict[str, np.ndarray] = {}
        for metric, cell in by_metric.items():
            layers = sorted({layer for layer, _ in cell})
            seeds = sorted({seed for _, seed in cell})
            if len(seeds) < 2:
                continue
            grids[metric] = np.array([[cell[(layer, seed)] for seed in seeds] for layer in layers])
        if not grids:
            continue
        # The composite average matches the main study: sWUGGY test all, sBLIMP test, tSC test.
        grids["test_avg"] = (grids["swuggy_test_all"] + grids["sblimp_test"] + grids["tsc_test"]) / 3.0
        studies[recipe] = grids
    return studies


def layers_in(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        names = [row[0] for row in csv.reader(handle) if row and row[0].startswith(DATA_ROW_PREFIX)]
    return sorted({int(n[len(DATA_ROW_PREFIX) :].split("-", 1)[0]) for n in names})


# ------------------------------------------------------------------------------- variance model


def mean_squared_offdiag_correlation(residuals: np.ndarray) -> float:
    """Mean squared off-diagonal correlation between layer residual vectors.

    ``residuals`` has shape (n_layers, n_seeds) and is already centred within each layer.
    Uses the raw squared sample correlation, which overstates rho^2 at small n and therefore
    understates ``df_eff`` -- the conservative direction.
    """
    n_layers = residuals.shape[0]
    if n_layers < 2:
        return 0.0
    corr = np.corrcoef(residuals)
    off_diagonal = ~np.eye(n_layers, dtype=bool)
    return float(np.mean(corr[off_diagonal] ** 2))


def effective_df(n_layers: int, n_seeds: int, mean_rho_squared: float) -> float:
    """Degrees of freedom of the layer-pooled variance, discounted for cross-layer correlation."""
    return n_layers * (n_seeds - 1) / (1.0 + (n_layers - 1) * mean_rho_squared)


@dataclass(frozen=True)
class PooledNoise:
    sigma: float
    df_eff: float
    bartlett_p: float
    mean_rho_squared: float
    per_layer_sigma: np.ndarray
    per_layer_mean: np.ndarray

    @property
    def homogeneous(self) -> bool:
        return self.bartlett_p >= ALPHA


def pooled_noise(grid: np.ndarray) -> PooledNoise:
    """Fit layer-specific means and one shared sigma to a (n_layers, n_seeds) grid."""
    n_layers, n_seeds = grid.shape
    per_layer_mean = grid.mean(axis=1)
    residuals = grid - per_layer_mean[:, None]
    per_layer_var = (residuals**2).sum(axis=1) / (n_seeds - 1)
    mean_rho_squared = mean_squared_offdiag_correlation(residuals)
    return PooledNoise(
        sigma=float(np.sqrt(per_layer_var.mean())),
        df_eff=effective_df(n_layers, n_seeds, mean_rho_squared),
        bartlett_p=float(stats.bartlett(*grid).pvalue),
        mean_rho_squared=mean_rho_squared,
        per_layer_sigma=np.sqrt(per_layer_var),
        per_layer_mean=per_layer_mean,
    )


def sigma_item(mean_accuracy_pct: float, n_pairs: int | None) -> float | None:
    """Binomial standard error of an accuracy over a fixed benchmark, in percentage points.

    This floor is a property of the benchmark, not of training: no number of seeds removes it.
    """
    if n_pairs is None:
        return None
    p = mean_accuracy_pct / 100.0
    return 100.0 * math.sqrt(p * (1.0 - p) / n_pairs)


@dataclass(frozen=True)
class NoiseFloor:
    key: str
    display: str
    n_pairs: int | None
    layers: list[int]
    slm: PooledNoise
    enc: PooledNoise
    sigma_total_ci: tuple[float, float]
    sigma_item: float | None
    sigma_repro: float | None = None

    @property
    def sigma_enc(self) -> float:
        """Measured directly by the encoder-seed grid; nothing is subtracted off it."""
        return self.enc.sigma

    @property
    def sigma_total(self) -> float:
        """Spread of a fully independent replication: both seeds redrawn."""
        return math.hypot(self.enc.sigma, self.slm.sigma)

    @property
    def dominance(self) -> float:
        """How many times larger the encoder-seed noise is than the SLM-seed noise."""
        return self.sigma_enc / self.slm.sigma if self.slm.sigma > 0 else float("inf")

    @property
    def homogeneous(self) -> bool:
        """Whether pooling sigma across layers is supported on *both* grids.

        sigma_tot combines both pooled variances, so a homogeneity failure on either grid
        propagates into it.
        """
        return self.enc.homogeneous and self.slm.homogeneous

    def threshold(self, k_a: int = 1, k_b: int = 1, alpha: float = ALPHA) -> float:
        """Smallest difference between two models that is significant at ``alpha``."""
        return _z(1.0 - alpha / 2.0) * self.sigma_total * math.sqrt(1.0 / k_a + 1.0 / k_b)

    def half_width(self, k: int = 1, alpha: float = ALPHA, n_slm_seeds: int = 1) -> float:
        """Calibrated error bar to print on a table cell backed by ``k`` encoder seeds.

        Built on se_of_mean, not sigma_tot / sqrt(k): the latter would ignore the shared pipeline
        seed and understate the bar for k > 1. The two coincide at k = 1, where se_of_mean reduces
        to sigma_tot exactly.
        """
        return _z(1.0 - alpha / 2.0) * self.se_of_mean(k, n_slm_seeds)

    def mdd(self, k_a: int = 1, k_b: int = 1, alpha: float = ALPHA, power: float = POWER) -> float:
        """Minimum difference detectable at ``power``, given ``k_a``/``k_b`` seeds per model."""
        crit = _z(1.0 - alpha / 2.0) + _z(power)
        return crit * self.sigma_total * math.sqrt(1.0 / k_a + 1.0 / k_b)

    def se_of_mean(self, k_enc: int, n_slm_seeds: int = 1) -> float:
        """Standard error of a mean over ``k_enc`` encoder seeds sharing ``n_slm_seeds`` LM seeds.

        NOT sigma_tot / sqrt(k_enc). Our 8-seed study runs every encoder seed through the *same*
        pipeline seed, so all eight share one weight initialisation: that draw is a common offset
        and never averages out, however many encoder seeds are added. The encoder term averages
        down, the language-model term does not.
        """
        return math.sqrt(self.sigma_enc**2 / k_enc + self.slm.sigma**2 / n_slm_seeds)

    def gap_to_beat(
        self,
        sigma_enc_other: float | None = None,
        k_enc: int = 8,
        n_slm_seeds: int = 1,
        alpha: float = ALPHA,
        power: float | None = None,
    ) -> float:
        """Gap a rival single-run model needs to clear to beat our ``k_enc``-seed mean.

        Their one run carries both components, sigma_enc_other^2 + sigma_SLM^2, while our mean
        carries sigma_enc^2 / k_enc + sigma_SLM^2. So sigma_SLM enters twice: once as our
        irreducible fixed-seed offset, once as their single draw. ``sigma_enc_other`` defaults to
        our own sigma_enc, which is the optimistic end -- SpidR 2.0 is the least noisy of the six
        recipes measured, so a rival is more likely to be noisier than not. ``power=None`` gives
        the significance threshold; set power to get the minimum detectable difference.
        """
        if sigma_enc_other is None:
            sigma_enc_other = self.sigma_enc
        var_ours = self.sigma_enc**2 / k_enc + self.slm.sigma**2 / n_slm_seeds
        var_theirs = sigma_enc_other**2 + self.slm.sigma**2
        crit = _z(1.0 - alpha / 2.0) + (0.0 if power is None else _z(power))
        return crit * math.sqrt(var_ours + var_theirs)

    def optimal_slm_runs_per_encoder(self, cost_ratio: float) -> float:
        """Optimal number of SLM runs per encoder given cost(encoder)/cost(SLM)."""
        if self.sigma_enc <= 0:
            return float("inf")
        return (self.slm.sigma / self.sigma_enc) * math.sqrt(cost_ratio)


def _sigma_total_ci(slm: PooledNoise, enc: PooledNoise, rng: np.random.Generator) -> tuple[float, float]:
    """Monte-Carlo CI for sigma_tot from the chi-square pivots of the two pooled variances.

    A nonparametric bootstrap over the 8 seed columns is tempting but biased here: resampling
    with replacement at n = 8 shrinks the variance by a factor (n-1)/n and produces
    implausibly tight upper bounds. The pivot ``df_eff * sigma_hat^2 / sigma^2 ~ chi2(df_eff)``
    is the same approximation the t-test already relies on, so the CI and the decision rule
    stay mutually consistent. sigma_tot^2 is now a *sum* of two such variances, so unlike the
    old difference form it cannot be driven to the zero boundary.
    """
    var_enc = enc.sigma**2 * enc.df_eff / rng.chisquare(enc.df_eff, N_BOOTSTRAP)
    var_slm = slm.sigma**2 * slm.df_eff / rng.chisquare(slm.df_eff, N_BOOTSTRAP)
    draws = np.sqrt(var_enc + var_slm)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_noise_floors(
    slm_study: dict[str, np.ndarray] | None = None,
    total_study: dict[str, np.ndarray] | None = None,
    layers: list[int] | None = None,
) -> dict[str, NoiseFloor]:
    slm_study = load_seed_study(SLM_SEED_CSV) if slm_study is None else slm_study
    total_study = load_seed_study(ENC_SEED_CSV) if total_study is None else total_study
    layers = layers_in(ENC_SEED_CSV) if layers is None else layers
    determinism = load_determinism_study()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    floors: dict[str, NoiseFloor] = {}
    for key, display, n_pairs in METRICS.values():
        if key not in slm_study or key not in total_study:
            continue
        slm_grid, enc_grid = slm_study[key], total_study[key]
        slm, enc = pooled_noise(slm_grid), pooled_noise(enc_grid)
        floors[key] = NoiseFloor(
            key=key,
            display=display,
            n_pairs=n_pairs,
            layers=layers,
            slm=slm,
            enc=enc,
            sigma_total_ci=_sigma_total_ci(slm, enc, rng),
            sigma_item=sigma_item(float(enc.per_layer_mean.mean()), n_pairs),
            sigma_repro=(
                float(determinism[key].std(ddof=1)) if key in determinism else None
            ),
        )
    return floors


@dataclass(frozen=True)
class RecipeSigma:
    recipe: str
    display: str
    normalised: bool
    n_seeds: int
    sigma_enc: dict[str, float]
    sigma_enc_raw: dict[str, float]
    peak_mean: dict[str, float]
    ratio_p: dict[str, float]


def recipe_sigmas(
    floors: dict[str, NoiseFloor] | None = None,
    studies: dict[str, dict[str, np.ndarray]] | None = None,
) -> list[RecipeSigma]:
    """sigma_tot per recipe, with an F-test of each against the SpidR 2.0 reference.

    These sheets pair one SLM run with each encoder seed, so what they measure is sigma_tot --
    the same quantity as the main study's encoder-seed grid, and not separable into its encoder
    and language-model parts. The F-test compares the two pooled variances; because layer
    residuals are correlated within a recipe, its p-values are optimistic and should be read as
    indicative of the size of the discrepancy rather than as exact.
    """
    floors = build_noise_floors() if floors is None else floors
    studies = load_recipe_studies() if studies is None else studies

    out: list[RecipeSigma] = []
    for recipe, (display, normalised) in RECIPES.items():
        if recipe not in studies:
            continue
        grids = studies[recipe]
        sigma, raw, peak, pvals = {}, {}, {}, {}
        for metric in RECIPE_TABLE_METRICS:
            fitted = pooled_noise(grids[metric])
            reference = floors[metric].enc
            raw[metric] = fitted.sigma
            # The F-test below stays on the raw pooled variances: subtracting a constant from both
            # sides is not an exact F pivot, and the correction is far smaller than the test's own
            # optimism from correlated layers.
            sigma[metric] = discount(fitted.sigma, floors[metric].sigma_repro)
            peak[metric] = float(fitted.per_layer_mean.max())
            f_stat = (fitted.sigma / reference.sigma) ** 2
            pvals[metric] = float(
                2.0
                * min(
                    stats.f.sf(f_stat, fitted.df_eff, reference.df_eff),
                    stats.f.cdf(f_stat, fitted.df_eff, reference.df_eff),
                )
            )
        out.append(
            RecipeSigma(
                recipe=recipe,
                display=display,
                normalised=normalised,
                n_seeds=grids[RECIPE_TABLE_METRICS[0]].shape[1],
                sigma_enc=sigma,
                sigma_enc_raw=raw,
                peak_mean=peak,
                ratio_p=pvals,
            )
        )
    return out


def sigma_by_normalisation(
    recipes: list[RecipeSigma],
    studies: dict[str, dict[str, np.ndarray]] | None = None,
    floors: dict[str, NoiseFloor] | None = None,
) -> dict[bool, dict[str, float]]:
    """sigma_enc pooled over the recipes that do and do not include the normalisation step."""
    studies = load_recipe_studies() if studies is None else studies
    floors = build_noise_floors() if floors is None else floors
    grouped: dict[bool, dict[str, float]] = {}
    for normalised in (False, True):
        members = [r.recipe for r in recipes if r.normalised is normalised]
        pooled: dict[str, float] = {}
        for metric in RECIPE_TABLE_METRICS:
            fits = [pooled_noise(studies[r][metric]) for r in members]
            weight = sum(f.df_eff for f in fits)
            combined = math.sqrt(sum(f.sigma**2 * f.df_eff for f in fits) / weight)
            pooled[metric] = discount(combined, floors[metric].sigma_repro)
        grouped[normalised] = pooled
    return grouped


# ------------------------------------------------------------------------------------ comparison


@dataclass(frozen=True)
class Comparison:
    metric: str
    delta: float
    p_value: float
    ci: tuple[float, float]
    rule: str
    significant: bool
    detail: dict[str, float] = field(default_factory=dict)


def _split_indices(n: int, n_a: int, rng: np.random.Generator) -> np.ndarray:
    """Group-A index matrix of shape (n_splits, n_a), enumerated exactly when feasible."""
    if math.comb(n, n_a) <= MAX_EXACT_PERMUTATIONS:
        return np.fromiter(
            (i for combo in combinations(range(n), n_a) for i in combo), dtype=int, count=math.comb(n, n_a) * n_a
        ).reshape(-1, n_a)
    return np.argsort(rng.random((MAX_EXACT_PERMUTATIONS, n)), axis=1)[:, :n_a]


def _permutation_p(a: np.ndarray, b: np.ndarray, splits: np.ndarray) -> float:
    """Two-sided permutation p-value for a difference in means, over precomputed splits."""
    pooled = np.concatenate([a, b])
    n_a, n = len(a), len(pooled)
    observed = a.mean() - b.mean()
    group_a = pooled[splits].sum(axis=1)
    null = group_a / n_a - (pooled.sum() - group_a) / (n - n_a)
    # +1 correction keeps the p-value valid (and non-zero) in the Monte-Carlo case.
    count = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    return min((count + 1) / (len(null) + 1), 1.0)


def permutation_is_attainable(n_a: int, n_b: int, alpha: float = ALPHA) -> bool:
    """Whether a two-sided permutation test on ``n_a`` vs ``n_b`` can reject at ``alpha`` at all.

    The most extreme observed split is tied with its mirror image, so the smallest attainable
    p-value is ``3 / (n_splits + 1)``. At 3 vs 3 that is 0.14: the test has exactly zero power
    and falling back to the calibrated threshold -- which borrows sigma and therefore does have
    power -- is strictly better than reporting a foregone "not significant".
    """
    n_splits = min(math.comb(n_a + n_b, n_a), MAX_EXACT_PERMUTATIONS)
    return 3.0 / (n_splits + 1) <= alpha


def _hodges_lehmann(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(a[:, None] - b[None, :]))


def _permutation_ci(a: np.ndarray, b: np.ndarray, splits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Invert the permutation test by bisecting on the shift applied to ``a``."""
    centre = _hodges_lehmann(a, b)
    span = max(4.0 * (a.std(ddof=1) + b.std(ddof=1)), 1e-6)

    def rejects(shift: float) -> bool:
        return _permutation_p(a - shift, b, splits) < alpha

    bounds = []
    for direction in (-1.0, 1.0):
        lo, hi = centre, centre + direction * span
        # The smallest attainable p-value is bounded below by 1/(n_splits+1); if even a huge
        # shift cannot be rejected, the interval is unbounded on this side.
        while not rejects(hi) and abs(hi - centre) < 1e4 * span:
            hi = centre + (hi - centre) * 2.0
        if not rejects(hi):
            bounds.append(direction * math.inf)
            continue
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if rejects(mid):
                hi = mid
            else:
                lo = mid
        bounds.append(lo)
    return min(bounds), max(bounds)


def compare(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    metric: str,
    floors: dict[str, NoiseFloor] | None = None,
    alpha: float = ALPHA,
    rng: np.random.Generator | None = None,
) -> Comparison:
    """Compare two models on one metric, choosing the rule from how many seeds each has.

    When the seed counts admit a permutation test that can actually reject at ``alpha`` (4 vs 4
    or better) the data speak for themselves and we use it. Otherwise -- the usual case, one
    encoder seed per model -- we fall back to the threshold calibrated on the SpidR 2.0 seed
    study, which borrows sigma from 8 seeds x 8 layers and therefore retains power.
    """
    floors = build_noise_floors() if floors is None else floors
    if metric not in floors:
        raise KeyError(f"unknown metric {metric!r}; known: {sorted(floors)}")
    floor = floors[metric]
    rng = np.random.default_rng(BOOTSTRAP_SEED) if rng is None else rng
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    delta = float(a.mean() - b.mean())

    if permutation_is_attainable(len(a), len(b), alpha):
        splits = _split_indices(len(a) + len(b), len(a), rng)
        p = _permutation_p(a, b, splits)
        return Comparison(
            metric=metric,
            delta=_hodges_lehmann(a, b),
            p_value=p,
            ci=_permutation_ci(a, b, splits, alpha),
            rule="permutation",
            significant=p < alpha,
            detail={"mean_difference": delta, "k_a": len(a), "k_b": len(b)},
        )

    se = floor.sigma_total * math.sqrt(1.0 / len(a) + 1.0 / len(b))
    p = float(2.0 * stats.norm.sf(abs(delta / se)))
    crit = _z(1.0 - alpha / 2.0)
    return Comparison(
        metric=metric,
        delta=delta,
        p_value=p,
        ci=(delta - crit * se, delta + crit * se),
        rule="calibrated",
        significant=abs(delta) > crit * se,
        detail={
            "threshold": crit * se,
            "sigma_total": floor.sigma_total,
            "k_a": len(a),
            "k_b": len(b),
        },
    )


def holm(p_values: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down: returns ``{name: rejected}``."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    n = len(ordered)
    rejected: dict[str, bool] = {}
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        still_rejecting = still_rejecting and p <= alpha / (n - rank)
        rejected[name] = still_rejecting
    return rejected


# --------------------------------------------------------------------------------- layer choice


def selection_bias(
    dev_metric: str = "dev_avg",
    test_metric: str = "test_avg",
    study: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Optimism incurred by picking the reported layer on dev instead of an oracle on test.

    For each encoder seed: take the layer that maximises ``dev_metric``, read ``test_metric``
    there, and compare against the best achievable ``test_metric``. The mean gap is the
    penalty paid for honest layer selection; the oracle gap is the bias you would incur by
    selecting on test directly.
    """
    study = load_seed_study(ENC_SEED_CSV) if study is None else study
    dev, test = study[dev_metric], study[test_metric]
    chosen = dev.argmax(axis=0)
    at_chosen = test[chosen, np.arange(test.shape[1])]
    best = test.max(axis=0)
    mean_per_layer = test.mean(axis=1)
    return {
        "mean_gap_to_oracle": float(np.mean(best - at_chosen)),
        "max_gap_to_oracle": float(np.max(best - at_chosen)),
        "oracle_selection_bias": float(np.mean(best) - mean_per_layer.max()),
        "n_distinct_layers_chosen": float(len(set(chosen.tolist()))),
    }


# ------------------------------------------------------------------------------------- reporting


def write_noise_floor_csv(floors: dict[str, NoiseFloor], out_path: Path = NOISE_FLOOR_CSV) -> Path:
    layers = next(iter(floors.values())).layers
    header = [
        "metric", "display", "primary", "n_pairs",
        *[f"mean_l{layer}" for layer in layers],
        "sigma_slm", "sigma_enc", "sigma_total", "sigma_total_lo", "sigma_total_hi", "sigma_item",
        "sigma_repro", "se_mean_k8", "ci95_mean_k8",
        "enc_over_slm", "bartlett_p_enc", "bartlett_p_slm",
        "homogeneous", "half_width_95_k1", "threshold_95_1v1",
        "mdd_k1", "mdd_k2", "mdd_k4", "mdd_k8",
    ]  # fmt: skip

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for floor in floors.values():
            writer.writerow([
                floor.key, floor.display, floor.key in PRIMARY_METRICS, floor.n_pairs or "",
                *[f"{m:.4f}" for m in floor.enc.per_layer_mean],
                f"{floor.slm.sigma:.4f}", f"{floor.sigma_enc:.4f}", f"{floor.sigma_total:.4f}",
                f"{floor.sigma_total_ci[0]:.4f}", f"{floor.sigma_total_ci[1]:.4f}",
                f"{floor.sigma_item:.4f}" if floor.sigma_item is not None else "",
                f"{floor.sigma_repro:.4f}" if floor.sigma_repro is not None else "",
                f"{floor.se_of_mean(8):.4f}", f"{floor.half_width(8):.4f}",
                f"{floor.dominance:.2f}",
                f"{floor.enc.bartlett_p:.4f}", f"{floor.slm.bartlett_p:.4f}",
                floor.homogeneous,
                f"{floor.half_width(1):.4f}", f"{floor.threshold():.4f}",
                *[f"{floor.mdd(k, k):.4f}" for k in (1, 2, 4, 8)],
            ])  # fmt: skip
    return out_path


def write_sigma_recipe_csv(
    floors: dict[str, NoiseFloor],
    recipes: list[RecipeSigma],
    grouped: dict[bool, dict[str, float]],
    out_path: Path = SIGMA_RECIPE_CSV,
) -> Path:
    header = ["recipe", "display", "normalised", "n_seeds", "reference"]
    for metric in RECIPE_TABLE_METRICS:
        header += [f"sigma_{metric}", f"sigma_raw_{metric}", f"p_vs_ref_{metric}"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        row: list[object] = ["spidr2", "SpidR 2.0", True, 8, True]
        for metric in RECIPE_TABLE_METRICS:
            discounted = discount(floors[metric].sigma_enc, floors[metric].sigma_repro)
            row += [f"{discounted:.4f}", f"{floors[metric].sigma_enc:.4f}", ""]
        writer.writerow(row)
        for recipe in recipes:
            row = [recipe.recipe, recipe.display, recipe.normalised, recipe.n_seeds, False]
            for metric in RECIPE_TABLE_METRICS:
                row += [
                    f"{recipe.sigma_enc[metric]:.4f}",
                    f"{recipe.sigma_enc_raw[metric]:.4f}",
                    f"{recipe.ratio_p[metric]:.4f}",
                ]
            writer.writerow(row)
        for normalised, pooled in grouped.items():
            row = ["", f"pooled: normalisation {'present' if normalised else 'absent'}", normalised, "", False]
            for metric in RECIPE_TABLE_METRICS:
                row += [f"{pooled[metric]:.4f}", "", ""]
            writer.writerow(row)
    return out_path


def print_recipe_sigmas(
    floors: dict[str, NoiseFloor], recipes: list[RecipeSigma], grouped: dict[bool, dict[str, float]]
) -> None:
    labels = [floors[m].display for m in RECIPE_TABLE_METRICS]
    print("sigma_enc by recipe (points), one SLM run per encoder seed.")
    print(f"{'recipe':<26}{'n':>3}" + "".join(f"{label:>16}" for label in labels))
    print("-" * (29 + 16 * len(labels)))
    print(
        f"{'SpidR 2.0 (reference)':<26}{8:>3}"
        + "".join(
            f"{discount(floors[m].sigma_enc, floors[m].sigma_repro):>16.3f}" for m in RECIPE_TABLE_METRICS
        )
    )
    for recipe in recipes:
        cells = "".join(
            f"{recipe.sigma_enc[m]:>12.3f}{'*' if recipe.ratio_p[m] < ALPHA else ' ':>4}"
            for m in RECIPE_TABLE_METRICS
        )
        print(f"{recipe.display[:26]:<26}{recipe.n_seeds:>3}{cells}")
    print()
    for normalised, pooled in grouped.items():
        label = f"pooled, norm {'present' if normalised else 'absent'}"
        print(f"{label:<26}{'':>3}" + "".join(f"{pooled[m]:>16.3f}" for m in RECIPE_TABLE_METRICS))
    print()
    print("* differs from the SpidR 2.0 reference at alpha = 0.05 (F-test on the pooled variances).")


def print_decomposition(floors: dict[str, NoiseFloor]) -> None:
    print(f"{'metric':<22}{'sd_SLM':>8}{'sd_enc':>8}{'sd_tot':>8}{'95% CI':>16}{'sd_item':>9}{'sd_rep':>8}{'bartP':>7}")
    print("-" * 78)
    for floor in floors.values():
        item = f"{floor.sigma_item:.3f}" if floor.sigma_item is not None else "--"
        repro = f"{floor.sigma_repro:.3f}" if floor.sigma_repro is not None else "--"
        ci = f"[{floor.sigma_total_ci[0]:.2f}, {floor.sigma_total_ci[1]:.2f}]"
        failed = [
            name
            for name, noise in (("encoder", floor.enc), ("SLM", floor.slm))
            if not noise.homogeneous
        ]
        flag = f"  <- {'/'.join(failed)} grid fails Bartlett, pooled sigma unreliable" if failed else ""
        print(
            f"{floor.display:<22}{floor.slm.sigma:>8.3f}{floor.sigma_enc:>8.3f}"
            f"{floor.sigma_total:>8.3f}{ci:>16}{item:>9}{repro:>8}{floor.enc.bartlett_p:>7.3f}{flag}"
        )
    print()
    print("sd_item is the benchmark's own binomial error. It cancels in model-vs-model comparisons")
    print("on a fixed pair set, so it feeds no threshold; it bounds task-level claims only.")
    print("sd_rep is the irreproducibility floor: 8 identical-seed runs, forked by feature extraction.")
    limited = [f.display for f in floors.values() if f.sigma_item is not None and f.sigma_item > f.sigma_total]
    if limited:
        print(f"Small pair sets (sd_item > sd_tot, task-level claims only): {', '.join(limited)}")


def print_power(floors: dict[str, NoiseFloor], cost_ratio: float) -> None:
    print(f"{'metric':<22}{'k=1':>8}{'k=2':>8}{'k=4':>8}{'k=8':>8}{'bar(k=1)':>10}{'SLM/enc*':>10}")
    print("-" * 74)
    for floor in floors.values():
        print(
            f"{floor.display:<22}" + "".join(f"{floor.mdd(k, k):>8.2f}" for k in (1, 2, 4, 8))
            + f"{floor.half_width(1):>10.2f}{floor.optimal_slm_runs_per_encoder(cost_ratio):>10.2f}"
        )
    print()
    print(f"Minimum detectable difference at alpha={ALPHA}, power={POWER:.0%}, k seeds per model.")
    print("bar(k=1) is the calibrated +/- to print on a single-seed cell; two cells differ")
    print("significantly only if their bars are separated by sqrt(2) x bar, not merely disjoint.")
    print(f"SLM/enc* is the cost-optimal number of SLM runs per encoder at cost(enc)/cost(SLM)={cost_ratio:g}.")


# ------------------------------------------------------------------------------------------ CLI


def load_determinism_study(path: Path | None = None) -> dict[str, np.ndarray]:
    """Scores of the identical-seed control runs, one row per run, as percentages.

    The pipeline retains two checkpoints per run (keep_best_n=1 plus keep_last_n=1) and evaluates
    both, so each run contributes two rows. We take the best-on-dev one, which is the checkpoint
    that is *not* the last step; runs where best and last coincide contribute their single row.
    """
    path = DETERMINISM_JSONL if path is None else path
    if not path.exists():
        return {}
    per_run: dict[str, dict[int, dict]] = {}
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            step = int(re.search(r"/step_(\d+)/", row["slm_ckpt"]).group(1))
            per_run.setdefault(row["run_name"].split("__")[0], {})[step] = row

    # With exactly {best, last} retained, the earlier step is the best-on-dev one.
    chosen = [steps[min(steps)] for steps in per_run.values()]

    study: dict[str, np.ndarray] = {}
    for field, key in DETERMINISM_FIELDS.items():
        if all(field in row for row in chosen):
            study[key] = np.array([row[field] * 100.0 for row in chosen])
    for split in ("dev", "test"):
        parts = [f"swuggy_{split}_invocab", f"swuggy_{split}_oov"]
        if all(p in study for p in parts):
            study[f"swuggy_{split}_all"] = sum(study[p] for p in parts) / len(parts)
    parts = ["swuggy_test_all", "sblimp_test", "tsc_test"]
    if all(p in study for p in parts):
        study["test_avg"] = sum(study[p] for p in parts) / len(parts)
    return study


def discount(sigma: float, sigma_repro: float | None) -> float:
    """Remove the irreproducibility floor from a sigma_enc estimate.

    Every sigma_enc, in the main grid and in each recipe study alike, contains one copy of the
    pipeline's own irreproducibility (feature extraction is not run under deterministic
    algorithms). sigma_repro is measured on SpidR 2.0 only and reused across recipes, which assumes
    it transfers -- reasonable for a cuDNN autotuning effect, but not verified per recipe. The
    correction is small: at most 0.02 points on any cell.
    """
    if not sigma_repro:
        return sigma
    return math.sqrt(max(0.0, sigma**2 - sigma_repro**2))


def worst_recipe_sigma_enc(metric: str, recipes: list[RecipeSigma]) -> tuple[float, str]:
    """Largest sigma_enc measured for ``metric`` among the non-reference recipes."""
    candidates = [(r.sigma_enc[metric], r.display) for r in recipes if metric in r.sigma_enc]
    return max(candidates)


def beat_table(
    floors: dict[str, NoiseFloor],
    recipes: list[RecipeSigma],
    k_enc: int = 8,
    alpha: float = ALPHA,
) -> list[dict[str, object]]:
    """Gap a single-run rival must clear to beat our k_enc-seed mean, bracketing their unknown sigma."""
    out: list[dict[str, object]] = []
    for metric in RECIPE_TABLE_METRICS:
        floor = floors[metric]
        worst, worst_name = worst_recipe_sigma_enc(metric, recipes)
        out.append({
            "metric": metric,
            "display": floor.display,
            "primary": metric in PRIMARY_METRICS,
            "se_ours": floor.se_of_mean(k_enc),
            "se_ours_naive": floor.sigma_total / math.sqrt(k_enc),
            "sigma_enc_worst": worst,
            "worst_recipe": worst_name,
            "gap_optimistic": floor.gap_to_beat(k_enc=k_enc, alpha=alpha),
            "gap_conservative": floor.gap_to_beat(worst, k_enc=k_enc, alpha=alpha),
            "mdd_optimistic": floor.gap_to_beat(k_enc=k_enc, alpha=alpha, power=POWER),
            "mdd_conservative": floor.gap_to_beat(worst, k_enc=k_enc, alpha=alpha, power=POWER),
        })  # fmt: skip
    return out


def print_beat(table: list[dict[str, object]], k_enc: int = 8) -> None:
    print(f"Gap a 1-run rival needs to beat our {k_enc}-encoder-seed mean (accuracy points).")
    print(f"{'metric':18}{'SE(ours)':>10}{'naive':>8}{'sig 95%':>18}{'MDD 80%':>18}")
    print(f"{'':18}{'':>10}{'':>8}{'opt.':>9}{'cons.':>9}{'opt.':>9}{'cons.':>9}")
    print("-" * 72)
    for row in table:
        print(
            f"{row['display']:18}{row['se_ours']:>10.2f}{row['se_ours_naive']:>8.2f}"
            f"{row['gap_optimistic']:>9.2f}{row['gap_conservative']:>9.2f}"
            f"{row['mdd_optimistic']:>9.2f}{row['mdd_conservative']:>9.2f}"
        )
    print()
    print("SE(ours) is sqrt(sigma_enc^2/k + sigma_SLM^2), not sigma_tot/sqrt(k): all k runs share")
    print("one pipeline seed, so that draw is a common offset and never averages out.")
    print("opt. assumes the rival is as stable as SpidR 2.0; cons. gives it the noisiest sigma_enc")
    print("measured across the other recipes -- their sigma is unknown and spans 2.5x.")


def write_beat_csv(table: list[dict[str, object]], out_path: Path = BEAT_CSV) -> Path:
    fields = [
        "metric", "display", "primary", "se_ours", "se_ours_naive", "sigma_enc_worst",
        "worst_recipe", "gap_optimistic", "gap_conservative", "mdd_optimistic", "mdd_conservative",
    ]  # fmt: skip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({
                k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()
            })  # fmt: skip
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    decompose = sub.add_parser("decompose", help="variance decomposition and noise-floor table")
    decompose.add_argument("-o", "--output", type=Path, default=NOISE_FLOOR_CSV)
    decompose.add_argument("--no-write", action="store_true", help="print only, do not write the CSV")

    power = sub.add_parser("power", help="minimum detectable difference vs number of seeds")
    power.add_argument("--cost-ratio", type=float, default=20.0, help="cost(encoder) / cost(SLM)")

    validate = sub.add_parser("validate", help="sigma_tot across the other 8-seed encoder studies")
    validate.add_argument("-o", "--output", type=Path, default=SIGMA_RECIPE_CSV)
    validate.add_argument("--no-write", action="store_true", help="print only, do not write the CSV")

    beat = sub.add_parser("beat", help="gap a 1-seed rival needs to beat our 8-seed mean")
    beat.add_argument("-k", "--k-enc", type=int, default=8, help="encoder seeds behind our mean")
    beat.add_argument("-o", "--output", type=Path, default=BEAT_CSV)
    beat.add_argument("--no-write", action="store_true", help="print only, do not write the CSV")

    cmp_parser = sub.add_parser("compare", help="test two models on one metric")
    cmp_parser.add_argument("--metric", required=True)
    cmp_parser.add_argument("--a", required=True, help="comma-separated scores for model A")
    cmp_parser.add_argument("--b", required=True, help="comma-separated scores for model B")
    cmp_parser.add_argument("--alpha", type=float, default=ALPHA)

    args = parser.parse_args()
    floors = build_noise_floors()

    if args.command == "decompose":
        print_decomposition(floors)
        print()
        bias = selection_bias()
        print(
            f"Layer selection on dev costs {bias['mean_gap_to_oracle']:.3f} pt of test avg. on average "
            f"(worst seed {bias['max_gap_to_oracle']:.3f}); selecting on test directly would inflate "
            f"the reported score by {bias['oracle_selection_bias']:.3f} pt."
        )
        if not args.no_write:
            print(f"\nWrote {write_noise_floor_csv(floors, args.output)}")
    elif args.command == "power":
        print_power(floors, args.cost_ratio)
    elif args.command == "beat":
        table = beat_table(floors, recipe_sigmas(floors, load_recipe_studies()), k_enc=args.k_enc)
        print_beat(table, args.k_enc)
        if not args.no_write:
            print(f"Wrote {write_beat_csv(table, args.output)}")
    elif args.command == "validate":
        studies = load_recipe_studies()
        recipes = recipe_sigmas(floors, studies)
        grouped = sigma_by_normalisation(recipes, studies, floors)
        print_recipe_sigmas(floors, recipes, grouped)
        if not args.no_write:
            print(f"\nWrote {write_sigma_recipe_csv(floors, recipes, grouped, args.output)}")
    else:
        a = [float(x) for x in args.a.split(",")]
        b = [float(x) for x in args.b.split(",")]
        result = compare(a, b, args.metric, floors=floors, alpha=args.alpha)
        verdict = "SIGNIFICANT" if result.significant else "not significant"
        print(f"{floors[args.metric].display}: delta = {result.delta:+.3f} pt  ({result.rule})")
        print(f"  {(1 - args.alpha):.0%} CI [{result.ci[0]:+.3f}, {result.ci[1]:+.3f}]   p = {result.p_value:.4f}   {verdict}")
        for name, value in result.detail.items():
            print(f"  {name} = {value:.4g}")


if __name__ == "__main__":
    main()
