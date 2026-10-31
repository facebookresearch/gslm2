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

See the [environment guide](environment.md) for dependency versions and CUDA setup.

Vocoder training requires CUDA and supports torchrun or Slurm with one process
per GPU. Duration training uses one GPU when available and otherwise CPU.
Inference supports CPU and CUDA.

## Prepare training data

Both models require **full-rate raw code IDs**, with `deduplicate: false` and
`cross_stream_deduplicate: false`. Frame repetitions provide duration targets
and preserve audio alignment. For six streams, keep layers 5–10 in order
throughout extraction and synthesis.

The encoder must produce a 320-sample unit hop at 16 kHz, as the default SpidR
strides do. Pooled units or other strides require a different vocoder.

### 1. Prepare source manifests

| Component | Source manifests |
|---|---|
| Duration prediction | [Libri-Light](data/librilight.md): small for training, large development split for validation |
| Unit vocoder | [LJSpeech](data/ljspeech.md): fixed train, valid, and test splits |
| Generative evaluation | [LibriSpeech](data/librispeech.md#generative-evaluation): fixed audio selections and prompt boundaries |

### 2. Extract units

Choose the matching [extraction template](../configs/misc/README.md): layer 7
for one stream or layers 5–10 for six streams. Set `checkpoint`, the manifests
under `targets`, and `output_folder`. For six streams:

```bash
# Run in gslm2; torchrun / Slurm extraction is also supported.
gslm2-extract-units --config \
  configs/misc/extract/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10-dup.yaml
```

### 3. Match outputs to training inputs

Outputs live under `output_folder/<layer>/`: `7/` for one stream and `5/` for
all six aligned streams.

| Output | Contents | Used by |
|---|---|---|
| `duration_predictor/<manifest_stem>_partial_<rank>.txt` | Full-rate units, chunked at 2,048 frames per stream | Duration trainer |
| `unit_vocoder/ljspeech_{train,valid,test}.jsonl` | Complete utterances with audio paths | Vocoder trainer |
| `generative_eval/full/<manifest_stem>.jsonl` | Complete evaluation utterances | Continuation generation |
| `generative_eval/prompt/<manifest_stem>.jsonl` | Independently encoded waveform prefixes | LM prompts |

Set the trainer paths to these outputs using the [path fields](../configs/misc/README.md#paths-to-edit).
Duration training and validation share `duration_predictor/`, filtered by
`small_segmented*.txt` and `large_segmented_60s_dev*.txt`. Duration targets are
derived from the full-rate rows. Vocoder training reads the train and valid
JSONLs directly, accepting `file_name` and `audio_repr`.

TXT shards remain separate by rank; JSONLs are merged across ranks. TXT rows
concatenate aligned streams in layer order; JSONL strings use `|` between streams.
See [Data formats](data/formats.md#duration-and-vocoder-inputs) for input conventions.

## Train

Use the matching [duration and vocoder configs](../configs/misc/README.md).
These are flat component training configs; the reference `duration_predictor`
and `unit_vocoder` YAML wrappers are also accepted. Vocoder worker counts use
the automatic default.

```bash
gslm2-train-duration configs/misc/duration/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.vocoder configs/misc/vocoder/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml
```

Slurm launchers keep the two jobs separate. Set their account, QoS, resource
requests, and Conda environment before submission:

```bash
sbatch scripts/launchers/slurm_duration.sbatch \
  configs/misc/duration/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml
sbatch scripts/launchers/slurm_vocoder.sbatch \
  configs/misc/vocoder/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml
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
  --duration-checkpoint /path/to/runs/duration_predictor/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10/best.pt \
  --vocoder-checkpoint /path/to/runs/unit_vocoder/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10/vocoder-47977 \
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
gslm2-resynthesize reconstructed-audio/ \
  /path/to/runs/unit_vocoder/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10/vocoder-47977 \
  /path/to/framewise-units/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10-dup/5/generative_eval/full/librispeech_test-clean_1k_min6s.jsonl
```

Vocoder model checkpoints retain the reference standalone unit-HiFi-GAN
format.

## Prepare other full-rate JSONL inputs

`gslm2-prepare-synthesis-data` converts other full-rate JSONL inputs into
canonical vocoder manifests and duration TXT shards:

```bash
gslm2-prepare-synthesis-data /path/to/extracted/train.jsonl \
  /path/to/full-rate-units --split train --num-streams 6
gslm2-prepare-synthesis-data /path/to/extracted/valid.jsonl \
  /path/to/full-rate-units --split valid --num-streams 6
```

It writes `train.jsonl`, `valid.jsonl`, and TXT shards under `train/` and
`valid/`. The preparation command refuses to overwrite an existing split.
Set the duration and vocoder config paths to these outputs when using this
conversion workflow.

Input JSONL accepts encoder fields `file_name` / `audio_repr`, or canonical
fields `audio` / `units`. Units may be a whitespace-delimited string, a list
of integers, or aligned nested lists; strings use `|` between streams.
