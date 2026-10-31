# Optional generative evaluation configs

The templates cover the optional continuation evaluation workflow for 350M
single-stream SLM and multi-stream GSLM2 experiments. Install the scoring
dependencies with the `eval` extra in the native runtime.

Use each YAML unchanged across decoding, synthesis, scoring, and judging.
Stage paths are derived from the same temperature and seed, including the judge input.

- `dev/`: seven SLM temperatures and six GSLM2 temperatures, each with ten
  seeds; 130 tasks in total.
- `test/`: ten seeds at reference calibrated temperatures, SLM 1.31 and
  GSLM2 1.58; 20 tasks in total.

See [Generative evaluation](../../../docs/generative-evaluation.md) for runtime
setup, data requirements, artifact formats, calibration, and launchers.
