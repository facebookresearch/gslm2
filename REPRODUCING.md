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
| VoxPopuli SLM / text LM | `configs/lm/recipes/{slm,textlm}/voxpopuli-*.yaml` | fairseq2 recipe CLI below |

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

Use `scripts/launchers/slurm_lm_steps.sbatch` for Slurm jobs (see below).
The following are the underlying recipe training and evaluation commands.

Prepare token shards (see `DATA.md`), set `dataset.path` in the recipe, then run
in the `gslm2_lm` environment installed with the `lm` extra (see
[installation guidance](README.md#installation)):

```bash
conda activate gslm2_lm
# The same recipe entry point handles SLM, textlm, and GSLM2 configs.
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.lm.recipes.train.run_recipe \
  --config-file configs/lm/recipes/slm/ll-6kh-300m.yaml runs/slm-ll-6kh
```

Validation and checkpoint selection use fairseq2's recipe regime. For benchmark
scoring, pass tokenized JSONL pairs (speech units or text tokenizer IDs):

```bash
gslm2-eval-slm --config-file configs/lm/recipes/slm/ll-6kh-300m.yaml \
  --checkpoint runs/slm-ll-6kh/checkpoints/step_25000 \
  --pairs-dir /path/to/tokenized-pairs \
  --benchmarks lexical_test_invocab lexical_test_oov syntactic_test sSC tSC \
  --out results/slm-ll-6kh.jsonl
```

The same evaluation command accepts `configs/lm/recipes/textlm/*.yaml` and
`configs/lm/recipes/gslm2/*.yaml`. Native LM checkpoints/configs from the previous layout
are no longer accepted by these LM commands.

Build audio pair manifests from ZeroSpeech downloads:

```bash
python scripts/create_slm_pair_manifests.py --data-dir /path/to/sLM21-dataset \
  --output-dir /path/to/sLM21-pairs
```

Unit extraction writes test JSONL manifests. Pair them with
`gslm2.lm.eval.create_pairs.create_paired_file`. Multi-stream units use
`|`-joined streams in evaluation pairs.

For the public experiments under `configs/lm`, replace the template paths
with your own and submit one config to run
preparation, training, and evaluation automatically:

```bash
sbatch --account=ACCOUNT --qos=QOS --nodes=4 \
  scripts/launchers/slurm_lm_steps.sbatch \
  configs/lm/gslm2/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m.yaml
```

See [LM experiments](configs/lm/README.md) for environments
and output paths. Standalone recipe configs run with the fairseq2 CLI above.

## Duration prediction and speech synthesis

See [SYNTHESIS.md](SYNTHESIS.md) for full-rate data preparation and independent
training using `configs/misc/duration/` and `configs/misc/vocoder/`, followed by
`gslm2-synthesize` to convert generated speech units to audio.
