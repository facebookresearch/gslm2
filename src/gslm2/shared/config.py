# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Job submission resource configuration."""

from typing import NotRequired, Required, TypedDict


class SlurmConfig(TypedDict):
    account: Required[str]
    nodes: Required[int]
    gpus_per_node: Required[int]
    qos: NotRequired[str]
    time: NotRequired[int]
    cpus_per_task: NotRequired[int]
    mem_per_gpu: NotRequired[str]
    constraint: NotRequired[str]
    partition: NotRequired[str]
    exclude: NotRequired[str]
