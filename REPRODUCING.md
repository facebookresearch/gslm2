# Reproducing the GSLM2 paper

This file maps each paper table/figure to the commands that produce it.

## Training

| Artifact | Config | Command |
|---|---|---|
| SpidR 2.0 encoder (LS-960h) | `configs/encoder/spidr2.0-ls960.yaml` | `gslm2-train-encoder -A <acct> -N 2 -G 8 -q <qos> --dump <dir> configs/encoder/spidr2.0-ls960.yaml` |
| SpidR 2.0 encoder (VoxPopuli En/Fr/De) | `configs/encoder/spidr2.0-voxpopuli-{en,fr,de}.yaml` | same, 2 nodes |
| Single-stream SLM | `configs/lm/slm/*.yaml` | LM steps launcher below |
| GSLM2 multi-stream SLM | `configs/lm/gslm2/*.yaml` | LM steps launcher below |
| Text LM | `configs/lm/textlm/*.yaml` | LM steps launcher below |

## Unit extraction (SLM / GSLM2 inputs)

Unit extraction accepts a user-provided YAML matching `FeatExtractConfig`
in `src/gslm2/encoder/units/config.py`. Set the encoder checkpoint, input
manifests, output folder, layers, vocabulary, and deduplication settings.

```bash
gslm2-extract-units --config /path/to/extraction.yaml
```

For single-stream SLM units, select one layer and enable `deduplicate`.
For six-stream GSLM2 units, use layers 5–10, `clustering: codes`,
`vocab_size: [256, 256, 256, 256, 256, 256]`, `deduplicate: false`,
`cross_stream_deduplicate: true`, and `chunk_size: 4096`.

## Encoder evaluation (`tab:encoder-eval`, `fig:frontend-eval`)

```bash
CKPT=/path/to/spidr2.0-seed0-final.pt
# ABX triphone (continuous + discrete), dev-clean / dev-other
gslm2-eval-abx $CKPT abx.csv dev-clean.tsv triphone-dev-clean.item spidr \
  --speaker_mode all --context_mode within --layers 10,11,12
gslm2-eval-abx $CKPT abx-discrete.csv dev-clean.tsv triphone-dev-clean.item spidr \
  --speaker_mode all --context_mode within --layers 10,11,12 --codebook_proba
# MAP word discovery
gslm2-eval-map $CKPT map.csv dev-clean-map.tsv dev-clean-words.jsonl spidr --layers 10,11,12
# Codebook quality: purity / PNMI / bitrate + P(phone|code) + durations
gslm2-eval-codebook $CKPT spidr dev-clean-map.tsv dev-clean.jsonl codebook-out/
# DiscoPhon cross-lingual (per language; eng shown)
gslm2-eval-discophon $CKPT spidr manifest-eng-test.csv alignment-eng-test.txt \
  triphone-eng-test.item discophon.csv eng --layers 5,6,7,8,9,10,11,12
# Stream similarity between codebook layers (Fig. stream similarity)
python scripts/analyze_codebook_pairwise_nmi.py --root codebook-out/ --split zr2021-dev-clean-map --out similarity/
# Layerwise plots
python scripts/plot_abx.py abx.csv -o abx.png
python scripts/plot_map.py map.csv -o map.png
python scripts/plot_pnmi_layerwise.py codebook-out/ -o pnmi.png
```

## LM training and evaluation (fairseq2)

Use the experiments under `configs/lm/{slm,gslm2,textlm}/`. Replace the
template paths with your own (see [LM configs](configs/lm/README.md)) and
install the `lm` extra in `gslm2_lm` using the
[installation guidance](README.md#installation).

Submit one config to prepare data, train and validate, and evaluate:

```bash
sbatch --account=ACCOUNT --qos=QOS --nodes=4 \
  scripts/launchers/slurm_lm_steps.sbatch \
  configs/lm/gslm2/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m.yaml
```

The launcher uses `gslm2_lm` for every stage. Its optional arguments
`START_STEP END_STEP RUN_NAME` let you train or evaluate a prepared run.
Validation and checkpoint selection use fairseq2's recipe regime; evaluation
scores the retained checkpoints on the configured benchmarks.

To run the stages directly, use the same experiment config and run name:

```bash
conda activate gslm2_lm
LM_CONFIG=configs/lm/slm/spidr2-slm_codes-l7_ll-6kh_350m.yaml
export SLM_RUN_NAME=SLM_example
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.lm.prepare --config_path "$LM_CONFIG"
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.lm.train --config_path "$LM_CONFIG"
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.lm.eval.benchmarks --config_path "$LM_CONFIG"
```

Preparation writes token shards and benchmark pairs. Multi-stream units use
`|`-joined streams in evaluation pairs. Speech benchmark pairing reads
ZeroSpeech labels from `slm.feature_extraction.slm21_root`.

Build audio pair manifests from ZeroSpeech downloads:

```bash
python scripts/create_slm_pair_manifests.py --data-dir /path/to/sLM21-dataset \
  --output-dir /path/to/sLM21-pairs
```

## Duration prediction and speech synthesis

See [SYNTHESIS.md](SYNTHESIS.md) for full-rate data preparation and independent
training using `configs/misc/duration/` and `configs/misc/vocoder/`, followed by
`gslm2-synthesize` to convert generated speech units to audio.
