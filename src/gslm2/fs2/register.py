# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer


# This function is called by fairseq2 during process startup to register
# extensions such as new model architectures or datasets. See the bottom of
# gslm2 repo's pyproject.toml where we set this function as a fairseq2
# extension.
def setup_fairseq2_extension(container: DependencyContainer) -> None:
    # The needed registrations are done in the recipe right now.
    # We keep this for clarity and possible future use.
    pass
