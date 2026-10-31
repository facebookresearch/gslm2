# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Score generated continuations with conditional GenPPL and S12."""

from gslm2.eval.generative.cli import run_stage


def main(argv: list[str] | None = None) -> int:
    """Score generated or existing candidate manifests."""
    return run_stage("score", argv)


if __name__ == "__main__":
    raise SystemExit(main())
