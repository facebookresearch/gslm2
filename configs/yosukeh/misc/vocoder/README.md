# SpidR 2.0 unit vocoders

These configs follow the SpidR 2.0 experiments in
`/storage/home/yosukeh/slm-adapt/metasr-internal/configs/yosukeh/gslm2/unit_vocoder`:

- `spidr2-slm_codes-l7.yaml`: single-stream layer-7 unit HiFi-GAN.
- `spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml`: six aligned streams from
  layers 5–10, with `concat_lin` embedding aggregation.

Training settings match the references: 256 raw codes per stream,
2,000 epochs, seed 1234, learning rate 0.0002, AdamW betas [0.8, 0.99],
gamma 0.999, batch size 64 per GPU, 8,960-sample crops, a 320-sample unit hop,
bfloat16 training, and validation every 5,000 steps. The 500,000-step cap
and automatic worker count match the reference config defaults.

## Data and outputs

The configs use our extracted LJSpeech JSONL files under
`/checkpoint/devai/yosukeh/gslm2_test/framewise_units/<extraction_name>/<layer>/unit_vocoder/`:

- Training: `ljspeech_train.jsonl`.
- Validation: `ljspeech_valid.jsonl`.

The trainer accepts the extraction fields `file_name` and `audio_repr`
directly. Keep full-rate, non-deduplicated raw IDs; multi-stream strings use
`|` between streams in layer order. Audio must be mono at 16 kHz.

Training checkpoints, inference vocoders, and metrics live under
`/checkpoint/devai/yosukeh/gslm2_test/unit_vocoder/<experiment_name>/`.
Training resumes from the latest checkpoint when present.

## Launch

Submit from the repository root using the native `gslm2` environment
installed with `.[native]` (see the repository README):

```bash
sbatch scripts/launchers/slurm_vocoder.sbatch \
  configs/yosukeh/misc/vocoder/spidr2-slm_codes-l7.yaml

sbatch scripts/launchers/slurm_vocoder.sbatch \
  configs/yosukeh/misc/vocoder/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml
```

The launcher requests eight GPUs, eight tasks, and 12 hours. Batch size 64
is per GPU. Override `--qos` and `--time` as needed.

The installed `gslm2_lm` environment has the audio stack but lacks `torchdata`,
which vocoder training requires. To use that environment, edit the launcher's
`conda activate` command and install `torchdata`:

```bash
conda activate gslm2_lm
uv pip install 'torchdata==0.11.0'

sbatch scripts/launchers/slurm_vocoder.sbatch \
  configs/yosukeh/misc/vocoder/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml
```

For reconstruction checks, use full-rate LibriSpeech evaluation units with
an exported `vocoder-<step>` directory:

```bash
gslm2-resynthesize /path/to/reconstructed-audio \
  /checkpoint/devai/yosukeh/gslm2_test/unit_vocoder/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10/vocoder-STEP \
  /checkpoint/devai/yosukeh/gslm2_test/framewise_units/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10-dup/5/generative_eval/full/librispeech_test-clean_1k_min6s.jsonl
```

Replace `vocoder-STEP` with an exported checkpoint from your run.
