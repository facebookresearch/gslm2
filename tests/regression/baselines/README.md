# Regression baselines

All baselines are CPU-generated and run in standard CI:
- `spidr_inference_output.pt` — SpidR encoder inference parity
- `slm_inference_output.pt` — SLM (SpidrOpt) inference parity
- `spidr_loss_trajectory.json` — SpidR encoder 50-step SSL loss parity
- `spidr_v2_loss_trajectory.json` — SpidR-v2 (sjepa) encoder 50-step SSL loss parity
- `slm_loss_trajectory.json` — SLM 50-step training loss parity

Regenerate after intentional model changes:

    python tests/regression/baselines/regen_baselines.py inference   # .pt files
    python tests/regression/baselines/regen_baselines.py loss        # .json trajectories
    python tests/regression/baselines/regen_baselines.py all         # everything

Loss baselines are reproduced from the exact tiny manifest the tests build
(`tests/regression/helpers.py::build_tiny_manifest`) and the same runners the
tests call, so the committed JSON matches what the test feeds in. Commit the
regenerated files in the same PR as the model change so the diff makes the
intent obvious.
