# Generative evaluation reference

See [Generative speech evaluation](../generative-evaluation.md) for setup and
commands to run replicas, score existing audio, and calibrate temperatures.

## Dependencies

Decoding uses fairseq2 in `gslm2_lm`. Synthesis, ASR, perplexity, and judging use
native PyTorch in `gslm2` with the `eval` extra.

The `eval` extra adds Accelerate, Transformers 5, Librosa, Torchvision, and
Whisper's English normalizer. Qwen3's audio processor requires Torchvision;
the native PyTorch pin selects a compatible build.

`local_files_only: true` requires downloaded weights; public templates use false.

## Paths and checkpoint loading

YAML paths resolve relative to the config directory; command-line paths use the
working directory. Prefix relative local scorer directories with `./` to
distinguish them from Hugging Face model IDs.

- `model_config`: a saved fairseq2 recipe or full experiment YAML. Loading
  preserves vocabulary offsets, model overrides, and Depformer order. If omitted,
  the loader searches ancestor directories for `config.yaml`.
- `model_checkpoint`: an exported state dictionary or fairseq2 step directory.
  Sharded checkpoints are consolidated into the replica's output directory.

## CLI overrides

Override a temperature, seed, or experiment output root on every affected stage:

```bash
gslm2-generate "$CONFIG" --temperature 1.0 --seed 3 \
  --output-directory /path/to/evaluation/my-experiment
```

The module command supports the same stages:

```bash
python -m gslm2.eval.generative "$CONFIG" --stage score
```

Valid stages are `decode`, `synthesize`, `score`, `judge`, and `calibrate`.

## Artifacts

Replica files are relative to `output_directory/t<TEMPERATURE>-p<TOP_P>/<SEED>/`,
using floating-point formatting such as `t1.0-p0.9/3`.

| Artifact | Contents |
|---|---|
| `decode/decoded_units.jsonl` | Deduplicated decoded units, full-rate prompt units, frame counts, and IDs |
| `generation/units.jsonl` | Full-rate vocoder input with the original prompt frame count |
| `generation/audio/*.flac` | Synthesized 16 kHz candidate speech |
| `generation/synthesized.csv` | Speech candidate IDs and audio paths |
| `scoring/transcripts.jsonl` | Continuation transcripts and conditioning prompts |
| `scoring/transcripts.metadata.json` | Input and transcript hashes for cache validation |
| `scoring/results.jsonl` | Per-candidate conditional GenPPL, repetition, and degeneracy |
| `scoring/summary.json` | Aggregate GenPPL, repetition, S12, and degeneracy |
| `judge/results.jsonl` | Judge inputs, feedback, and scores |
| `judge/summary.json` | Aggregate judge scores and counts |
| `config.json` | Resolved replica settings; each stage also records its settings |

## Continuation protocol

Stages preserve manifest order across ranks and require matching, unique IDs.
Prompts are deduplicated jointly across streams for LM conditioning. Duration
prediction uses the entire decoded context, then expands the continuation.
The unchanged full-rate prompt is prepended; the continuation is capped at ten seconds.

Maximum decoding steps follow the reference heuristic:
`min(int(continuation_frames * prompt_dedup_frames / prompt_frames * 1.5),
continuation_frames)`. Short continuations are not padded.

## Candidate and reference manifests

Candidate CSVs and JSONLs accept `id`/`fileid` and `audio_path`/`path`/`audio`.
References accept the bundled boundary format or
explicit `prompt_text`, `reference_continuation`, and `prompt_end_seconds` fields.

## ASR and transcript reuse

Qwen3 transcribes audio after the saved prompt boundary. Conditioning uses
reference prompt text; if alignment is false or the text contains `<unk>`,
the candidate's prompt audio is transcribed too.

Cache validation checks audio, candidate/reference manifests, ASR settings, and
transcripts. Caches without metadata are regenerated. Set
`evaluation.reuse_transcripts: false` to force transcription.

## Metrics

GenPPL uses Llama-3.2-1B to score the complete continuation conditioned on the
prompt, masking prompt and padding tokens. Scoring applies Whisper English
normalization and numeral expansion. Inputs over `max_length` raise an error.

Repetition uses the first 15 normalized continuation words. R1 counts every
occurrence of repeated unigrams; R2 does the same for adjacent bigrams:

`S12 = 0.5 * (mean(candidate R1) / mean(reference R1) + mean(candidate R2) / mean(reference R2))`.

References use the same split's boundary manifest. Summaries include finite
GenPPL mean, median, and count; repetition means; S12; and the fraction of
continuations with fewer than five words. Empty continuations have null GenPPL
and contribute to repetition and degeneracy.

## Calibration

Calibration averages per-seed repetition means and median GenPPL values. It
interpolates temperature linearly in S12 and GenPPL in log space, targets S12=1,
and rounds temperature to two decimal places. Missing replicas, inconsistent
references/windows, or an unbracketed target raise errors.

Reference test temperatures for the 350M models are SLM 1.31 and GSLM2 1.58,
with GSLM2 Depformer annealing 0.5. Sweep sizes are listed in the
[config guide](../../configs/misc/generative_eval/README.md).

## Slurm settings

The launcher uses `gslm2_lm` for decoding and `gslm2` for native stages. It
prepends the active environment's `lib/` to `LD_LIBRARY_PATH`; set the same path
for direct native runs. Edit the launcher's Conda path and `conda activate`
commands for another installation.

For the approximately 93 GB Prometheus checkpoint, reduce concurrent loads:

```bash
sbatch --account=ACCOUNT --qos=QOS --nodes=2 --gpus-per-node=2 \
  --ntasks-per-node=2 --mem=256G scripts/launchers/slurm_generative.sbatch \
  "$CONFIG" judge
```
