# Model zoo

## SpidR 2.0 encoder

The released encoder has approximately **90M parameters** and was trained on
**LibriSpeech-960h, seed 0**.

Download the [checkpoint](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/spidr-2.0/final.pt)
and [configuration](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/spidr-2.0/config.json).
Keep `config.json` beside `final.pt` when loading the encoder. Package details
are provided in the [model card](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/spidr-2.0/README.md)
and [manifest](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/spidr-2.0/manifest.json).

Use the [native environment](environment.md#native-environment) and follow
[speech-unit extraction](reproducing.md#unit-extraction-slm--gslm2-inputs).
In the extraction config, set `checkpoint` to the downloaded `final.pt` and
`model_type` to `spidr`.

## Spoken language models

The spoken language models use units from the encoder above. **SLM** uses
layer 7; **GSLM** uses six aligned streams from layers 5–10, in that order.

| Model | Training data | Nominal size | Actual parameters | Downloads |
|---|---|---|---:|---|
| SLM | Libri-Light 600h | 125M | 87,026,688 | [Weights](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-slm_codes-l7_ll-600h_125m/model.pt) · [Config](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-slm_codes-l7_ll-600h_125m/config.yaml) |
| SLM | Libri-Light 6000h | 350M | 304,939,008 | [Weights](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-slm_codes-l7_ll-6kh_350m/model.pt) · [Config](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-slm_codes-l7_ll-6kh_350m/config.yaml) |
| SLM | Libri-Light 60000h | 1B | 1,213,857,792 | [Weights](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-slm_codes-l7_ll-60kh_1b/model.pt) · [Config](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-slm_codes-l7_ll-60kh_1b/config.yaml) |
| GSLM | Libri-Light 600h | 125M | 96,309,504 | [Weights](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-600h_125m/model.pt) · [Config](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-600h_125m/config.yaml) |
| GSLM | Libri-Light 6000h | 350M | 315,472,128 | [Weights](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m/model.pt) · [Config](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m/config.yaml) |
| GSLM | Libri-Light 60000h | 1B | 1,229,392,128 | [Weights](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-60kh_1b/model.pt) · [Config](https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2/checkpoints/lm/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-60kh_1b/config.yaml) |
| GSLM | 60M h conversational speech | 7B | | Planned release |

The size labels follow nominal OPT categories; actual parameter counts differ
between SLM and GSLM. Each released `model.pt` is the best retained checkpoint
according to validation `nll_loss`. Its training step and selection score are
recorded in `manifest.json`.

### Download and verify

Download all package files and verify their checksums:

```bash
GSLM2_BASE_URL=https://dl.fbaipublicfiles.com/shared/devai/assets/gslm2
MODEL=spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m
MODEL_DIR="/path/to/checkpoints/$MODEL"
mkdir -p "$MODEL_DIR"
for filename in model.pt config.yaml units.json README.md manifest.json LICENSE SHA256SUMS; do
  curl -fL "$GSLM2_BASE_URL/checkpoints/lm/$MODEL/$filename" \
    -o "$MODEL_DIR/$filename" || exit 1
done
(cd "$MODEL_DIR" && sha256sum -c SHA256SUMS)
```

For the encoder, use `checkpoints/spidr-2.0/` and download `final.pt`,
`config.json`, `README.md`, `manifest.json`, `LICENSE`, and `SHA256SUMS`.
Verify from the downloaded encoder directory with `sha256sum -c SHA256SUMS`.

### Loading and evaluation

Use the [LM environment](environment.md#lm-environment) with fairseq2 0.7 and
PyTorch 2.9, the versions used to verify the released packages.

Load `model.pt` with the matching `config.yaml`. Each package's `README.md`
provides Python loading, paired benchmark evaluation, and fine-tuning examples.
For data preparation and training, use a full
[LM experiment template](../configs/lm/README.md).

For [generative evaluation](generative-evaluation.md), set
`speech.model_checkpoint` to the downloaded `model.pt` and
`speech.model_config` to its matching `config.yaml`. Follow the guide to prepare
prompts and calibrate generation temperature on the development split.

Duration and unit-vocoder checkpoints are not included in this release. To
train the models required for audio generation, follow
[speech synthesis](synthesis.md).

## External baselines used in the paper

- [SpidR official checkpoint](https://github.com/facebookresearch/spidr).

## License

Released checkpoints are distributed under CC-BY-NC-4.0. See the `LICENSE`
in each package and the [repository license](../LICENSE).
