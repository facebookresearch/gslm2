# Experiment configs

See [Installation](../README.md#installation) for environments and command conventions.

## Encoder

See [Encoder experiments](encoder/README.md) for SpidR 2.0 templates on
LibriSpeech and VoxPopuli, including data preparation and launch instructions.

## Language models (fairseq2)

See [LM experiments](lm/README.md) for single-stream SLM, multi-stream GSLM2,
and text LM templates, with launcher instructions.

## Speech synthesis

- `misc/extract/` — full-rate single- and six-stream extraction for vocoder and duration training.
- `misc/duration/` — single- and six-stream duration prediction from full-rate units.
- `misc/vocoder/` — single- and six-stream unit HiFi-GAN training.

See [Speech synthesis](../docs/synthesis.md) for data preparation, training, and inference.

## Optional generative evaluation

See [Generative evaluation configs](misc/generative_eval/README.md) for dev/test
templates and [Generative evaluation](../docs/generative-evaluation.md) for
setup and commands.
