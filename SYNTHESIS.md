# Duration prediction and unit vocoding

`gslm2.duration` and `gslm2.vocoder` train independently with native PyTorch.
`gslm2.synthesis` connects their inference APIs. The vocoder implements
unit-conditioned HiFi-GAN; duration prediction supports Conformer and Transformer
encoders. Both support one stream or six aligned streams.

## Environment

```bash
conda activate gslm2
uv pip install -e '.[native]'
```

Use matching CUDA PyTorch/torchaudio builds for your machine. The `native`
extra uses PyTorch 2.13.0, TorchCodec 0.14.0, torchaudio >=2.11.0, and
`torchdata`. LM training and evaluation use the separate `gslm2_lm`
environment. See [installation guidance](README.md#installation).

Vocoder training requires CUDA and supports torchrun or Slurm with one process
per GPU. Duration training uses one GPU when available and otherwise CPU.
Inference supports CPU and CUDA. ASR/MOS evaluators are separate from these
components.

## Prepare training data

Both models require **full-rate, non-deduplicated raw code IDs**. LM shards
that have already been deduplicated cannot supply duration targets or aligned
vocoder training audio. For multi-stream units, preserve encoder-layer order
(layers 5, 6, 7, 8, 9, 10) throughout extraction and synthesis.

Use `configs/misc/extract/single-stream.yaml` for SpidR layer 7, or
`configs/misc/extract/multi-stream.yaml` for aligned layers 5–10. These configs
write JSONL for every split with deduplication and chunking disabled. Set the
encoder checkpoint, input manifests, and output folder before launching.

The encoder must have a 320-sample unit hop at 16 kHz. The default SpidR
convolution strides have this hop; pooled units or another encoder stride
require a different vocoder and are not supported here.

```bash
# Run in gslm2; torchrun / Slurm extraction is also supported.
gslm2-extract-units --config configs/misc/extract/multi-stream.yaml

# Run in gslm2.
gslm2-prepare-synthesis-data /path/to/framewise-units/multi-stream/5/train/train.jsonl \
  /path/to/full-rate-units --split train --num-streams 6
gslm2-prepare-synthesis-data /path/to/framewise-units/multi-stream/5/valid/valid.jsonl \
  /path/to/full-rate-units --split valid --num-streams 6
```

The merged JSONL filename follows the input CSV stem. Single-stream output
uses the selected layer directory, e.g. `7/`. The preparation command refuses
to overwrite an existing split. It produces:

```text
full-rate-units/
  train.jsonl         # audio paths plus full-rate units for the vocoder
  valid.jsonl
  train/shard_0.txt   # full-rate units for duration target construction
  valid/shard_0.txt
```

Input JSONL accepts encoder fields `file_name` / `audio_repr`, or canonical
fields `audio` / `units`. Units may be a whitespace-delimited string, a list
of integers, or aligned nested lists; strings use `|` between streams. Duration
TXT rows concatenate complete streams in order. Duration training derives
deduplicated units and run lengths from these full-rate rows.

## Train

Replace paths in `configs/misc/duration/{single-stream,multi-stream}.yaml` and
`configs/misc/vocoder/{single-stream,multi-stream}.yaml`. These are flat component
training configs; the reference `duration_predictor` and `unit_vocoder` YAML
wrappers are also accepted.

```bash
gslm2-train-duration configs/misc/duration/multi-stream.yaml
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.vocoder configs/misc/vocoder/multi-stream.yaml
```

Slurm launchers keep the two jobs separate. Set their account, QoS, resource
requests, and Conda environment before submission:

```bash
sbatch scripts/launchers/slurm_duration.sbatch configs/misc/duration/multi-stream.yaml
sbatch scripts/launchers/slurm_vocoder.sbatch configs/misc/vocoder/multi-stream.yaml
```

The duration trainer writes `step_N.pt`, `best.pt`, and `last.pt` checkpoints
containing `model` and `config`. The vocoder writes resumable `checkpoint-N.pt`
training states and inference directories `vocoder-N/` with `config.json` and
`model.safetensors`. Vocoder training automatically resumes its latest training
state in the configured work directory. Duration training starts a fresh run.

## Generate audio

Supply a JSONL manifest of deduplicated speech-unit sequences, in canonical
stream order. Its `audio` field names each output utterance; the original audio
need not exist. Output relative paths are derived from their common parent,
and the generated files use `.wav`.

```bash
gslm2-synthesize generated-units.jsonl generated-audio/ \
  --duration-checkpoint /path/to/duration/best.pt \
  --vocoder-checkpoint /path/to/vocoder/vocoder-50000 \
  --unit-offset 2
```

Use `--unit-offset 2` for speech-unit LM token IDs, and `0` for raw code IDs.
IDs below the offset are removed before subtracting the offset. Additional
special IDs can be supplied with `--special-token-ids`. Special-token frames
must be aligned across streams; mismatched frames are rejected. Text LM token
IDs are not valid speech-unit inputs.

The duration model predicts one duration per aligned tuple, and all streams
are expanded with that same duration. The vocoder produces 320 samples per
expanded frame at 16 kHz. To reconstruct audio from already full-rate units,
skip duration prediction:

```bash
gslm2-resynthesize reconstructed-audio/ /path/to/vocoder/vocoder-50000 \
  /path/to/full-rate-units/valid.jsonl
```

Vocoder model checkpoints retain the reference standalone unit-HiFi-GAN
format. Its MIT license is included at `src/gslm2/vocoder/LICENSE`.
