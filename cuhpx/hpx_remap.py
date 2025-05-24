# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cuhpx_remap
import torch


def is_power_of_two(n):
    return (n > 0) and (n & (n - 1)) == 0


def ring2nest(data: torch.Tensor, nside: int):

    if not is_power_of_two(nside):
        raise ValueError("nside must be a power of two")

    if data.dim() == 1:
        return cuhpx_remap.ring2nest(data, nside, data.size(-1))
    else:
        return cuhpx_remap.ring2nest_batch(data, nside, data.size(-1))


def nest2ring(data: torch.Tensor, nside: int):

    if not is_power_of_two(nside):
        raise ValueError("nside must be a power of two")

    if data.dim() == 1:
        return cuhpx_remap.nest2ring(data, nside, data.size(-1))
    else:
        return cuhpx_remap.nest2ring_batch(data, nside, data.size(-1))


def nest2flat(data: torch.Tensor, origin: str, clockwise: bool, nside: int):

    if not is_power_of_two(nside):
        raise ValueError("nside must be a power of two")

    if data.dim() == 1:
        return cuhpx_remap.nest2xy(data, origin, clockwise, nside, data.size(-1))
    else:
        return cuhpx_remap.nest2xy_batch(data, origin, clockwise, nside, data.size(-1))


def flat2nest(data: torch.Tensor, origin: str, clockwise: bool, nside: int):

    if not is_power_of_two(nside):
        raise ValueError("nside must be a power of two")

    if data.dim() == 1:
        return cuhpx_remap.xy2nest(data, origin, clockwise, nside, data.size(-1))
    else:
        return cuhpx_remap.xy2nest_batch(data, origin, clockwise, nside, data.size(-1))


def ring2flat(data: torch.Tensor, origin: str, clockwise: bool, nside: int):

    if not is_power_of_two(nside):
        raise ValueError("nside must be a power of two")

    if data.dim() == 1:
        return cuhpx_remap.ring2xy(data, origin, clockwise, nside, data.size(-1))
    else:
        return cuhpx_remap.ring2xy_batch(data, origin, clockwise, nside, data.size(-1))


def flat2ring(data: torch.Tensor, origin: str, clockwise: bool, nside: int):

    if not is_power_of_two(nside):
        raise ValueError("nside must be a power of two")

    if data.dim() == 1:
        return cuhpx_remap.xy2ring(data, origin, clockwise, nside, data.size(-1))
    else:
        return cuhpx_remap.xy2ring_batch(data, origin, clockwise, nside, data.size(-1))


def flat2flat(
    data: torch.Tensor, src_origin: str, src_clockwise: bool, dest_origin: str, dest_clockwise: bool, nside: int
):

    if not is_power_of_two(nside):
        raise ValueError("nside must be a power of two")

    if data.dim() == 1:
        return cuhpx_remap.xy2xy(data, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, data.size(-1))
    else:
        return cuhpx_remap.xy2xy_batch(
            data, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, data.size(-1)
        )
