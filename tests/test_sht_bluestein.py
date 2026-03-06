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

"""Tests for Bluestein FFT-based SHT implementation."""

import pytest
import torch
from conftest import get_bluestein_tol, get_roundtrip_tol

from cuhpx import SHT, iSHT


@pytest.mark.cuda
class TestSHTBluestein:
    """Test Bluestein FFT-based spherical harmonic transforms."""

    @pytest.fixture
    def signal(self, device, nside, dtype):
        """Create a random test signal."""
        npix = 12 * nside**2
        return torch.randn(npix, dtype=dtype, device=device)

    @pytest.fixture
    def sht_torch(self, device, nside, lmax, mmax):
        """Create standard SHT transform."""
        return SHT(nside, lmax=lmax, mmax=mmax, quad_weights="ring").to(device)

    @pytest.fixture
    def isht_torch(self, device, nside, lmax, mmax):
        """Create standard iSHT transform."""
        return iSHT(nside, lmax=lmax, mmax=mmax).to(device)

    @pytest.fixture
    def sht_bluestein(self, device, nside, lmax, mmax):
        """Create Bluestein SHT transform."""
        return SHT(nside, lmax=lmax, mmax=mmax, quad_weights="ring", use_bluestein=True).to(device)

    @pytest.fixture
    def isht_bluestein(self, device, nside, lmax, mmax):
        """Create Bluestein iSHT transform."""
        return iSHT(nside, lmax=lmax, mmax=mmax, use_bluestein=True).to(device)

    def test_sht_bluestein_matches_standard(self, signal, sht_torch, sht_bluestein):
        """Test that Bluestein SHT produces same results as standard SHT."""
        coeff_torch = sht_torch(signal.clone())
        coeff_bluestein = sht_bluestein(signal.clone())

        # Cast to same dtype for comparison (standard SHT may use different precision)
        coeff_bluestein = coeff_bluestein.to(coeff_torch.dtype)

        # Use Bluestein-specific tolerances (algorithm-limited, not precision-limited)
        rtol, atol = get_bluestein_tol("sht")
        assert torch.allclose(
            coeff_torch, coeff_bluestein, rtol=rtol, atol=atol
        ), f"SHT Bluestein/standard mismatch: max diff = {(coeff_torch - coeff_bluestein).abs().max():.2e}"

    def test_isht_bluestein_matches_standard(self, signal, sht_torch, isht_torch, isht_bluestein):
        """Test that Bluestein iSHT produces same results as standard iSHT."""
        coeff = sht_torch(signal)

        signal_torch = isht_torch(coeff.clone())
        signal_bluestein = isht_bluestein(coeff.clone())

        # Cast to same dtype for comparison (standard iSHT may use different precision)
        signal_bluestein = signal_bluestein.to(signal_torch.dtype)

        # Use Bluestein-specific tolerances (algorithm-limited, not precision-limited)
        rtol, atol = get_bluestein_tol("isht")
        assert torch.allclose(
            signal_torch, signal_bluestein, rtol=rtol, atol=atol
        ), f"iSHT Bluestein/standard mismatch: max diff = {(signal_torch - signal_bluestein).abs().max():.2e}"

    def test_bluestein_roundtrip(self, device, nside, lmax, mmax, dtype, complex_dtype, sht_bluestein, isht_bluestein):
        """Test SHT -> iSHT roundtrip with Bluestein implementation on bandlimited signal."""
        # Create bandlimited signal
        torch.manual_seed(42)
        coeffs = torch.zeros((lmax, mmax), dtype=complex_dtype, device=device)
        max_l = min(lmax // 2, nside // 2)
        for l_idx in range(max_l):
            for m_idx in range(min(l_idx + 1, mmax)):
                coeffs[l_idx, m_idx] = torch.randn(1, dtype=dtype).item() + 1j * torch.randn(1, dtype=dtype).item()

        signal = isht_bluestein(coeffs)
        coeff_back = sht_bluestein(signal)
        signal_back = isht_bluestein(coeff_back)

        # Roundtrip error is algorithm-limited, not precision-limited
        rtol, atol = get_roundtrip_tol()
        assert torch.allclose(
            signal_back, signal, rtol=rtol, atol=atol
        ), f"Bluestein roundtrip failed: max diff = {(signal_back - signal).abs().max():.2e}"
