# Reproducing the GSLM2 paper

This file maps each paper table/figure to the commands that produce it.

## Training

| Artifact | Config | Command |
|---|---|---|
| SpidR 2.0 encoder (LS-960h) | `configs/encoder/spidr2.0-ls960.yaml` | [Encoder launcher](../configs/encoder/README.md#launch) |
| SpidR 2.0 encoder (VoxPopuli En/Fr/De) | `configs/encoder/spidr2.0-voxpopuli-{en,fr,de}.yaml` | [Encoder launcher](../configs/encoder/README.md#launch) |
| Single-stream SLM | `configs/lm/slm/*.yaml` | [LM launcher](../configs/lm/README.md#launch) |
| GSLM2 multi-stream SLM | `configs/lm/gslm2/*.yaml` | [LM launcher](../configs/lm/README.md#launch) |
| Text LM | `configs/lm/textlm/*.yaml` | [LM launcher](../configs/lm/README.md#launch) |

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

These commands run directly (no Slurm needed) once the checkpoint and
label inputs exist. See [Encoder evaluation labels](data/encoder-eval-labels.md)
for the audio manifests, ABX `.item` download, and the word/phone
alignment files.

```bash
CKPT=/path/to/runs/spidr2.0/final.pt
EVAL_MANIFEST_DIR=/path/to/eval_manifests
# ABX triphone (continuous + discrete), dev-clean / dev-other
gslm2-eval-abx "$CKPT" abx.csv "$EVAL_MANIFEST_DIR/dev-clean.csv" \
  "$EVAL_MANIFEST_DIR/items/triphone-dev-clean.item" spidr \
  --speaker_mode all --context_mode within --layers 10,11,12
gslm2-eval-abx "$CKPT" abx-discrete.csv "$EVAL_MANIFEST_DIR/dev-clean.csv" \
  "$EVAL_MANIFEST_DIR/items/triphone-dev-clean.item" spidr \
  --speaker_mode all --context_mode within --layers 10,11,12 --codebook_proba
# MAP word discovery
gslm2-eval-map "$CKPT" map.csv "$EVAL_MANIFEST_DIR/dev-clean.csv" \
  "$EVAL_MANIFEST_DIR/dev-clean-words.jsonl" spidr --layers 10,11,12
# Codebook quality: purity / PNMI / bitrate + P(phone|code) + durations
gslm2-eval-codebook "$CKPT" spidr "$EVAL_MANIFEST_DIR/dev-clean.csv" \
  "$EVAL_MANIFEST_DIR/dev-clean.jsonl" codebook-out/
# DiscoPhon cross-lingual (per language; eng shown)
gslm2-eval-discophon $CKPT spidr manifest-eng-test.csv alignment-eng-test.txt \
  triphone-eng-test.item discophon.csv eng --layers 5,6,7,8,9,10,11,12
```

## LM training and evaluation (fairseq2)

Choose the matching experiment under `configs/lm/{slm,gslm2,textlm}/`.
The [LM guide](../configs/lm/README.md) covers path setup, launching,
stage selection, resuming, outputs, and [direct commands](../configs/lm/README.md#run-stages-directly).

For benchmark data, follow [ZeroSpeech and StoryCloze preparation](data/lm-benchmarks.md)
or [LT-Swap preparation](data/ltswap.md). Optional audio pair exports are documented
in [SLM Pair Manifest Format](../scripts/PAIR_MANIFEST_FORMAT.md).

## Duration prediction and speech synthesis

See [Speech synthesis](synthesis.md) for full-rate data preparation and independent
training using `configs/misc/duration/` and `configs/misc/vocoder/`, followed by
`gslm2-synthesize` to convert generated speech units to audio.

## Optional generative evaluation

Use `configs/misc/generative_eval/{dev,test}/` for single-stream SLM and multi-stream
GSLM2, with the decoding, continuation synthesis, scoring, and judge commands in
[Generative evaluation](generative-evaluation.md). Dev sweeps calibrate
temperature at S12=1; test configs use the reference calibrated 350M settings,
1.31 for SLM and 1.58 for GSLM2.
