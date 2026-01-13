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

"""Tests for data remapping operations (ring2nest, nest2ring) against healpy reference."""

import healpy as hp
import pytest
import torch

import cuhpx as hpx


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.int32, torch.float64])
def test_ring2nest_matches_healpy(device, nside, dtype):
    """Test that ring2nest produces identical results to healpy."""
    npix = 12 * nside**2
    tensor_ring = torch.arange(npix, device=device, dtype=dtype)

    # cuHPX ring2nest
    result_hpx = hpx.ring2nest(tensor_ring, nside).cpu()

    # healpy reference
    result_healpy = torch.tensor(
        hp.pixelfunc.reorder(tensor_ring.cpu().numpy(), inp="RING", out="NESTED"),
        dtype=dtype,
    )

    assert torch.equal(result_hpx, result_healpy), f"ring2nest mismatch for nside={nside}, dtype={dtype}"


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.int32, torch.float64])
def test_nest2ring_matches_healpy(device, nside, dtype):
    """Test that nest2ring produces identical results to healpy."""
    npix = 12 * nside**2
    tensor_nest = torch.arange(npix, device=device, dtype=dtype)

    # cuHPX nest2ring
    result_hpx = hpx.nest2ring(tensor_nest, nside).cpu()

    # healpy reference
    result_healpy = torch.tensor(
        hp.pixelfunc.reorder(tensor_nest.cpu().numpy(), inp="NESTED", out="RING"),
        dtype=dtype,
    )

    assert torch.equal(result_hpx, result_healpy), f"nest2ring mismatch for nside={nside}, dtype={dtype}"


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ring2nest_nest2ring_roundtrip(device, nside, dtype):
    """Test that ring2nest followed by nest2ring recovers the original data."""
    npix = 12 * nside**2
    original = torch.randn(npix, device=device, dtype=dtype)

    # Round trip: ring -> nest -> ring
    nested = hpx.ring2nest(original, nside)
    recovered = hpx.nest2ring(nested, nside)

    assert torch.equal(original, recovered), f"Round-trip failed for nside={nside}, dtype={dtype}"


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nest2ring_ring2nest_roundtrip(device, nside, dtype):
    """Test that nest2ring followed by ring2nest recovers the original data."""
    npix = 12 * nside**2
    original = torch.randn(npix, device=device, dtype=dtype)

    # Round trip: nest -> ring -> nest
    ring = hpx.nest2ring(original, nside)
    recovered = hpx.ring2nest(ring, nside)

    assert torch.equal(original, recovered), f"Round-trip failed for nside={nside}, dtype={dtype}"
