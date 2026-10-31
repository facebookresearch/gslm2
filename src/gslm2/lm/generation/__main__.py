# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate speech-unit or text continuations."""

from gslm2.eval.generative.cli import run_stage


def main(argv: list[str] | None = None) -> int:
    """Run the decoding stage in the LM environment."""
    return run_stage("decode", argv)


if __name__ == "__main__":
    raise SystemExit(main())
