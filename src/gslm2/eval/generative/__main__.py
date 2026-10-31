# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run a selected generative evaluation stage."""

from gslm2.eval.generative.cli import run_stage


def main(argv: list[str] | None = None) -> int:
    """Run decoding, synthesis, scoring, judging, or calibration."""
    return run_stage(None, argv)


if __name__ == "__main__":
    raise SystemExit(main())
