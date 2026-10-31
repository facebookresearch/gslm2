# Optional old-checkpoint generative evaluation

These configs translate
`metasr-internal-main/configs/yosukeh/gslm2/gen_eval_old_ckpt/` into this
repository's shared generative evaluation format. Each YAML is used for decoding,
continuation synthesis, scoring, and judging.

- `dev/` covers seven SLM temperatures and six GSLM2 temperatures with seeds
  0, 1, and 2; the current configs expand to 39 tasks.
- `test/` uses the currently configured temperatures, SLM 1.31 and GSLM2 1.58,
  with ten seeds each; 20 tasks in total. GSLM2 retains Depformer annealing 0.5.
- `topline/` uses the source dev-clean inputs and synthesis checkpoints with
  ground-truth continuations. These local oracle configs include a text oracle,
  which skips continuation synthesis.
- `models/` contains inference recipes derived from each checkpoint's saved
  `config.yaml`. They retain model preset, vocabulary, token offset, model
  overrides, and Depformer order. Training-only options from the old recipes are
  omitted.

The configs use the original unit manifests, LM/duration/vocoder checkpoints,
Qwen3, Llama, and Prometheus model paths. The text oracle also supplies its
tokenizer path. Scorer loading uses local
files only. The source generation TSV contains paths to `gen_eval/`; these
configs use its temperature values with the `gen_eval_old_ckpt` YAML inputs.

## Run one replica

From the repository root:

```bash
EVAL_CONFIG=configs/yosukeh/misc/generative_eval/test/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_350m.yaml

conda activate gslm2_lm
gslm2-generate "$EVAL_CONFIG"

conda activate gslm2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
gslm2-synthesize-continuations "$EVAL_CONFIG"
gslm2-eval-generative "$EVAL_CONFIG"
gslm2-eval-judge "$EVAL_CONFIG"
```

Run these commands on a GPU allocation.
For Slurm, submit one stage with:

```bash
sbatch --account=devai --qos=h200_devai_high \
  scripts/launchers/slurm_generative.sbatch "$EVAL_CONFIG" decode
```

Submit `synthesize` and `score` after their preceding stages finish. For `judge`,
add `--nodes=2 --gpus-per-node=2 --ntasks-per-node=2 --mem=256G`.

Outputs use:

```text
/checkpoint/devai/yosukeh/gslm2_test/gen_eval_old_ckpt/
  <split>/<model>/t<TEMPERATURE>-p0.9/<seed>/
    decode/
    generation/
    scoring/
    judge/
```

For example, the default GSLM2 test replica is at
`test/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_350m/t1.58-p0.9/0/`.
Oracle configs use `<split>/<model>/oracle/`.

## Sweeps and calibration

```bash
conda activate gslm2
TASK_COUNT=$(python -m gslm2.eval.generative.sweep \
  configs/yosukeh/misc/generative_eval/dev/*.yaml \
  --output build/gen_eval/old_ckpt_dev_tasks.jsonl)

sbatch --account=devai --qos=h200_devai_high --array="0-$((TASK_COUNT - 1))" \
  scripts/launchers/slurm_generative_array.sbatch \
  build/gen_eval/old_ckpt_dev_tasks.jsonl decode
```

Reuse this 39-task plan for decoding, synthesis, and scoring. The planner derives
the array size from the configured temperatures and seeds. Use Slurm dependencies
between stages.

After completing each model's dev sweep:

```bash
gslm2-calibrate-generative configs/yosukeh/misc/generative_eval/dev/CONFIG.yaml
```

See the [generative evaluation guide](../../../../docs/generative-evaluation.md)
for dependencies, artifact formats, and scoring existing candidate manifests.
