# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Judge semantic quality in post-ASR continuation text."""

from gslm2.eval.generative.cli import run_stage


def main(argv: list[str] | None = None) -> int:
    """Run the Prometheus continuation judge."""
    return run_stage("judge", argv)


if __name__ == "__main__":
    raise SystemExit(main())
