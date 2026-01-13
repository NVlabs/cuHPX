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

"""Tests for batched remapping operations."""

import pytest
import torch

import cuhpx


@pytest.mark.cuda
def test_batched_ring2nest_matches_single(device, nside):
    """Test that batched ring2nest produces same results as individual operations."""
    npix = 12 * nside**2
    m, n = 4, 8
    signal = torch.randn(m, n, npix, dtype=torch.float32, device=device)

    # Batched operation
    result_batch = cuhpx.ring2nest(signal, nside)

    # Single operations
    result_single = torch.zeros_like(result_batch)
    for i in range(m):
        for j in range(n):
            result_single[i, j, :] = cuhpx.ring2nest(signal[i, j, :], nside)

    assert torch.equal(
        result_batch, result_single
    ), f"Batched ring2nest doesn't match single operations for nside={nside}"


@pytest.mark.cuda
def test_batched_nest2ring_matches_single(device, nside):
    """Test that batched nest2ring produces same results as individual operations."""
    npix = 12 * nside**2
    m, n = 4, 8
    signal = torch.randn(m, n, npix, dtype=torch.float32, device=device)

    # Batched operation
    result_batch = cuhpx.nest2ring(signal, nside)

    # Single operations
    result_single = torch.zeros_like(result_batch)
    for i in range(m):
        for j in range(n):
            result_single[i, j, :] = cuhpx.nest2ring(signal[i, j, :], nside)

    assert torch.equal(
        result_batch, result_single
    ), f"Batched nest2ring doesn't match single operations for nside={nside}"


@pytest.mark.cuda
@pytest.mark.parametrize("batch_shape", [(2,), (3, 4), (2, 3, 5)])
def test_batched_remap_various_shapes(device, nside, batch_shape):
    """Test batched remapping with various batch dimensions."""
    npix = 12 * nside**2
    signal = torch.randn(*batch_shape, npix, dtype=torch.float32, device=device)

    # Round trip should recover original
    nested = cuhpx.ring2nest(signal, nside)
    recovered = cuhpx.nest2ring(nested, nside)

    assert torch.equal(signal, recovered), f"Round-trip failed for batch_shape={batch_shape}, nside={nside}"
