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

"""Tests for spherical harmonic transforms (SHT/iSHT) accuracy."""

import pytest
import torch
from conftest import get_roundtrip_tol

from cuhpx import SHT, iSHT


@pytest.mark.cuda
def test_sht_isht_bandlimited_roundtrip(device, nside, lmax, mmax, dtype, complex_dtype):
    """Test that SHT followed by iSHT recovers a band-limited signal.

    When a signal is strictly band-limited to lmax, the SHT->iSHT roundtrip
    should recover it with high accuracy.
    """
    sht = SHT(nside, lmax=lmax, mmax=mmax, quad_weights="ring").to(device)
    isht = iSHT(nside, lmax=lmax, mmax=mmax).to(device)

    # Create strictly band-limited signal via iSHT of known coefficients
    # Only populate coefficients up to a fraction of lmax for strict band-limiting
    coeffs = torch.zeros((lmax, mmax), dtype=complex_dtype, device=device)
    max_l = min(lmax // 2, nside // 2)  # Use half of lmax for safety margin
    torch.manual_seed(42)
    for l_idx in range(max_l):
        for m_idx in range(min(l_idx + 1, mmax)):
            coeffs[l_idx, m_idx] = torch.randn(1, dtype=dtype).item()

    # Generate signal from coefficients
    signal = isht(coeffs)

    # Reconstruct via SHT + iSHT
    reconstructed = isht(sht(signal))

    # Roundtrip error is algorithm-limited, not precision-limited
    rtol, atol = get_roundtrip_tol()
    assert torch.allclose(
        reconstructed, signal, rtol=rtol, atol=atol
    ), f"Bandlimited roundtrip failed: max diff = {(reconstructed - signal).abs().max():.2e}"


@pytest.mark.cuda
def test_sht_output_shape(device, nside, lmax, mmax, dtype):
    """Test that SHT produces output with correct shape."""
    npix = 12 * nside**2
    sht = SHT(nside, lmax=lmax, mmax=mmax, quad_weights="ring").to(device)

    signal = torch.randn(npix, dtype=dtype, device=device)
    coeffs = sht(signal)

    assert coeffs.shape == (lmax, mmax), f"Expected shape ({lmax}, {mmax}), got {coeffs.shape}"
    assert coeffs.is_complex(), "SHT output should be complex"


@pytest.mark.cuda
def test_isht_output_shape(device, nside, lmax, mmax, complex_dtype):
    """Test that iSHT produces output with correct shape."""
    npix = 12 * nside**2
    isht = iSHT(nside, lmax=lmax, mmax=mmax).to(device)

    coeffs = torch.randn(lmax, mmax, dtype=complex_dtype, device=device)
    signal = isht(coeffs)

    assert signal.shape == (npix,), f"Expected shape ({npix},), got {signal.shape}"


@pytest.mark.cuda
def test_sht_isht_consistency(device, nside, lmax, mmax, dtype, complex_dtype):
    """Test that SHT and iSHT are consistent inverses for bandlimited signals."""
    sht = SHT(nside, lmax=lmax, mmax=mmax, quad_weights="ring").to(device)
    isht = iSHT(nside, lmax=lmax, mmax=mmax).to(device)

    # Start with coefficients and verify roundtrip
    torch.manual_seed(42)
    coeffs = torch.zeros((lmax, mmax), dtype=complex_dtype, device=device)
    max_l = min(lmax // 2, nside // 2)
    for l_idx in range(max_l):
        for m_idx in range(min(l_idx + 1, mmax)):
            coeffs[l_idx, m_idx] = torch.randn(2, dtype=dtype).sum()

    # coeffs -> signal -> coeffs_back
    signal = isht(coeffs)
    coeffs_back = sht(signal)

    # Roundtrip error is algorithm-limited, not precision-limited
    rtol, atol = get_roundtrip_tol()
    assert torch.allclose(
        coeffs_back[:max_l, :max_l], coeffs[:max_l, :max_l], rtol=rtol, atol=atol
    ), f"Coefficient roundtrip failed: max diff = {(coeffs_back[:max_l, :max_l] - coeffs[:max_l, :max_l]).abs().max():.2e}"
