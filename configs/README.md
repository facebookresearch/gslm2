# Training and extraction configs

Template paths are `/path/to/...` placeholders. Replace them before launching.

## Encoder

- `encoder/` — native SpidR 2.0 training on LibriSpeech and VoxPopuli.

Launch encoder training with `gslm2-train-encoder`.

## Language models (fairseq2)

Public LM experiments live under [`lm/`](lm/README.md): single-stream SLM,
multi-stream GSLM2, and text LM templates with preparation, training, and
evaluation settings. Replace `/path/to/...` with your own paths and pass one
config to the LM steps launcher.

[`yosukeh/lm/`](yosukeh/lm/README.md) contains end-to-end experiments derived
from Yosukeh's internal GSLM2 configs, with preparation settings and dataset
paths included. The public experiments use the same settings with editable
paths.

## Speech synthesis

- `misc/extract/` — full-rate single- and six-stream extraction for vocoder and duration training.
- `misc/duration/` — single- and six-stream duration prediction from full-rate units.
- `misc/vocoder/` — single- and six-stream unit HiFi-GAN training.

See [SYNTHESIS.md](../SYNTHESIS.md) for data preparation, training, and inference.
