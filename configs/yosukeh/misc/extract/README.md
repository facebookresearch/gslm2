# Framewise unit extraction

These standalone configs follow the three experiments in
`/storage/home/yosukeh/metasr-internal/metasr-internal-main/configs/yosukeh/gslm2/framewise_units`:

- `spidr2-slm_codes-l7-dup.yaml`: SpidR 2.0 layer-7 codes.
- `spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10-dup.yaml`: six aligned SpidR 2.0 streams, layers 5–10.
- `spidr-slm_kmeans-l6-dup.yaml`: SpidR layer-6 features clustered with a pretrained 256-center k-means model.

The `-dup` suffix means frame repetitions are retained. Both deduplication
settings are disabled. Checkpoints and input manifests match the references;
outputs are rooted at
`/checkpoint/devai/yosukeh/gslm2_test/framewise_units/<config_stem>/`.
Set `kmeans.km_path` before running the k-means experiment: the reference also
contains a placeholder, and extraction requires the fitted layer-6 model.

## Prepare manifests

- **Duration prediction:** use the
  [Libri-Light training and validation manifests](../../../../docs/data/librilight.md).
- **Unit vocoder:** follow
  [LJSpeech preparation](../../../../docs/data/ljspeech.md)
  to resample audio and create the fixed train, valid, and test manifests.
- **Generative evaluation:** use the
  [bundled LibriSpeech selections and prompt boundaries](../../../../docs/data/librispeech.md#generative-evaluation)
  and the helper to resolve paths against your local download.

Replace the corresponding `targets` paths with the generated local manifests
before running extraction.

## Run

Use an environment with the encoder and audio dependencies installed
(`gslm2` with the `native` extra; see the repository README).

```bash
gslm2-extract-units --config \
  configs/yosukeh/misc/extract/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10-dup.yaml
```

For extraction on eight GPUs, use the same config with:

```bash
torchrun --standalone --nproc_per_node=8 -m gslm2.encoder.units --config \
  configs/yosukeh/misc/extract/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10-dup.yaml
```

To reserve eight GPUs and run extraction as a Slurm batch job, submit from
the repository root:

```bash
sbatch scripts/launchers/slurm_extract_units.sbatch \
  configs/yosukeh/misc/extract/spidr2-slm_codes-l7-dup.yaml
```

The launcher requests one node, eight GPUs, eight tasks, and 12 hours. It
activates `gslm2_lm`. To use the native `gslm2` environment, edit the launcher's
`conda activate` command. Slurm options such as
`--qos`, `--nodes`, and `--time` can override the launcher defaults.
Logs are written to `slurm/unit-extract_<job_id>.out` and `.err`.

These are flat extraction configs for `gslm2-extract-units`, not end-to-end LM
configs for `slurm_lm_steps.sbatch`. They run extraction without training or
benchmark pairing.

## Outputs

Each config routes manifests by downstream component, matching the references:

- `<layer>/duration_predictor/<manifest_stem>_partial_<rank>.txt`: Libri-Light
  small and development data, chunked at 2048 frames per stream. Each row
  concatenates aligned streams in layer order.
- `<layer>/unit_vocoder/<manifest_stem>.jsonl`: LJSpeech train, valid, and test
  units, with one complete utterance per row and its audio filename.
- `<layer>/generative_eval/full/<manifest_stem>.jsonl`: complete LibriSpeech
  dev-clean and test-clean evaluation utterances.
- `<layer>/generative_eval/prompt/<manifest_stem>.jsonl`: independently encoded
  waveform prefixes using the corresponding 3-second prompt boundary manifests.

`<layer>` is `7` for single-stream codes, `5` for all six aligned streams,
and `6` for k-means. JSONL rows use `file_name` and `audio_repr`; multi-stream
strings separate streams with `|`. JSONL is not chunked. Rank-local JSONL
shards are merged after each manifest; TXT shards remain separate.

Prompt waveforms are cropped before normalization and bidirectional encoding,
then padded only as needed to produce the boundary's `prompt_frames`. They
are not obtained by slicing full-utterance units. All outputs retain raw code
IDs without the LM's offset of 2.
