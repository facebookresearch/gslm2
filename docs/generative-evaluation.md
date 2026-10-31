# Generative speech evaluation

Generate speech continuations, then score their linguistic quality with ASR,
conditional GenPPL, repetition, and LLM judging. The [configs](../configs/misc/generative_eval/README.md)
cover 350M single-stream SLMs and multi-stream GSLM2.

## Prerequisites

Install `gslm2_lm` and `gslm2` using the [environment guide](environment.md).
Add the scoring dependencies to `gslm2`:

```bash
conda activate gslm2
uv pip install -e '.[native,eval]'
```

- Prepare the [LibriSpeech evaluation manifests](data/librispeech.md#generative-evaluation)
  and [full-rate utterance and prompt units](synthesis.md#prepare-training-data).
  Keep the fixed selections, prompt boundaries, and stream order.
- Have LM, duration predictor, and vocoder checkpoints ready. Scorers accept
  model IDs or local directories; the Llama model may require Hugging Face access.

## Run one replica

Choose a test config and set input manifests and checkpoints under `speech`,
plus `output_directory`. Check scorer locations under `evaluation` and `judge`.

Run on a GPU allocation using the same config for every stage:

```bash
CONFIG=configs/misc/generative_eval/test/spidr2-slm_codes-l7_ll-6kh_350m.yaml

conda activate gslm2_lm
gslm2-generate "$CONFIG"

conda activate gslm2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
gslm2-synthesize-continuations "$CONFIG"
gslm2-eval-generative "$CONFIG"
gslm2-eval-judge "$CONFIG"
```

Each command runs one temperature/seed combination. Apply any
[overrides](generative-evaluation/reference.md#cli-overrides) consistently across
stages.

### Results

See the [artifact reference](generative-evaluation/reference.md#artifacts) for
output directories, generated audio, metric summaries, and per-candidate results.

### Submit with Slurm

```bash
sbatch --account=ACCOUNT --qos=QOS scripts/launchers/slurm_generative.sbatch \
  "$CONFIG" score
```

Supported stages are `decode`, `synthesize`, `score`, and `judge`. The launcher
selects the environment and sets `LD_LIBRARY_PATH`. See [Slurm settings](generative-evaluation/reference.md#slurm-settings)
for changing the environment names and judge resource requests.

## Score existing audio

Use `gslm2` with `.[native,eval]` and the same library path shown above.
A scoring config needs references and scorer settings; it can omit `speech`:

```yaml
mode: speech
reference_manifest: ./prompt_boundaries.jsonl
output_directory: ./evaluation
evaluation:
  qwen3:
    model_path: /path/to/Qwen3-Omni-30B-A3B-Instruct
  perplexity:
    model_path: /path/to/Llama-3.2-1B
  s12:
    window_words: 15
```

```bash
gslm2-eval-generative scoring.yaml --candidate-manifest synthesized.csv
gslm2-eval-judge scoring.yaml --input-manifest /path/to/post-asr/results.jsonl
```

Point `--input-manifest` to the `scoring/results.jsonl` written by the scoring
command. Candidates and references must have matching, unique IDs. See
[candidate and reference formats](generative-evaluation/reference.md#candidate-and-reference-manifests)
and [transcript reuse](generative-evaluation/reference.md#asr-and-transcript-reuse).

## Sweep temperatures and calibrate

Use the dev configs to plan temperature/seed combinations, then submit decoding:

```bash
conda activate gslm2
TASK_COUNT=$(python -m gslm2.eval.generative.sweep \
  configs/misc/generative_eval/dev/*.yaml --output build/gen_eval/dev_tasks.jsonl)

sbatch --account=ACCOUNT --qos=QOS --array="0-$((TASK_COUNT - 1))" \
  scripts/launchers/slurm_generative_array.sbatch \
  build/gen_eval/dev_tasks.jsonl decode
```

Submit `synthesize` and then `score` using the same task manifest and array range.
Use `--dependency=afterok:JOB_ID` to wait for the preceding stage. The planner
reports the task count; [config details](../configs/misc/generative_eval/README.md)
list the supplied sweep sizes.

After scoring all dev replicas, run calibration once per model:

```bash
gslm2-calibrate-generative configs/misc/generative_eval/dev/CONFIG.yaml
```

Calibration writes `output_directory/calibration.json`, targeting S12=1.
Use its temperature for newly trained models. The test templates use reference
calibrated settings; see [calibration details](generative-evaluation/reference.md#calibration).

## Reference

[Evaluation reference](generative-evaluation/reference.md) covers checkpoint
loading, continuation generation, metric definitions, and developer checks.
Paired likelihood benchmarks use `gslm2-eval-slm`.
