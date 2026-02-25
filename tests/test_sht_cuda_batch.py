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

"""Tests for batched SHT/iSHT CUDA operations."""

import pytest
import torch

from cuhpx import SHTCUDA, iSHTCUDA


@pytest.mark.cuda
class TestSHTCUDABatch:
    """Test batched spherical harmonic transforms on CUDA."""

    @pytest.fixture
    def batch_dims(self):
        """Return batch dimensions (m, n) for testing."""
        return (2, 3)

    @pytest.fixture
    def batch_signal(self, device, nside, batch_dims, dtype):
        """Create a batched random signal on device."""
        m, n = batch_dims
        npix = 12 * nside**2
        return torch.randn(m, n, npix, dtype=dtype, device=device)

    @pytest.fixture
    def sht(self, nside, lmax, mmax):
        """Create SHT transform."""
        return SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights="ring")

    @pytest.fixture
    def isht(self, nside, lmax, mmax):
        """Create iSHT transform."""
        return iSHTCUDA(nside, lmax=lmax, mmax=mmax)

    def test_sht_batch_matches_single(self, device, nside, batch_signal, sht, batch_dims, dtype):
        """Test that batched SHT matches element-wise SHT."""
        m, n = batch_dims

        # Batched transform
        coeff_batch = sht(batch_signal)

        # Element-wise transform
        coeff_single = torch.zeros_like(coeff_batch)
        for i in range(m):
            for j in range(n):
                coeff_single[i, j, :] = sht(batch_signal[i, j, :])

        # Compare results - should be identical
        # Use tighter tolerances for float64
        rtol = 1e-5 if dtype == torch.float32 else 1e-10
        atol = 1e-4 if dtype == torch.float32 else 1e-10
        assert torch.allclose(
            coeff_batch, coeff_single, rtol=rtol, atol=atol
        ), f"SHT batch/single mismatch: max diff = {(coeff_batch - coeff_single).abs().max():.2e}"

    def test_isht_batch_matches_single(self, device, nside, batch_signal, sht, isht, batch_dims, dtype):
        """Test that batched iSHT matches element-wise iSHT."""
        m, n = batch_dims

        # Get coefficients first
        coeff = sht(batch_signal)

        # Batched inverse transform
        signal_batch = isht(coeff)

        # Element-wise inverse transform
        signal_single = torch.zeros_like(signal_batch)
        for i in range(m):
            for j in range(n):
                signal_single[i, j, :] = isht(coeff[i, j, :])

        # Compare results - should be identical
        # Use tighter tolerances for float64
        rtol = 1e-5 if dtype == torch.float32 else 1e-10
        atol = 1e-6 if dtype == torch.float32 else 1e-10
        assert torch.allclose(
            signal_batch, signal_single, rtol=rtol, atol=atol
        ), f"iSHT batch/single mismatch: max diff = {(signal_batch - signal_single).abs().max():.2e}"

    @pytest.mark.parametrize("batch_shape", [(4,), (2, 2), (2, 2, 2)])
    def test_various_batch_shapes(self, device, nside_small, batch_shape, dtype):
        """Test batched SHT with various batch dimensions."""
        npix = 12 * nside_small**2
        lmax = 3 * nside_small - 1
        mmax = lmax

        signal = torch.randn(*batch_shape, npix, dtype=dtype, device=device)

        sht = SHTCUDA(nside_small, lmax=lmax, mmax=mmax, quad_weights="ring")
        isht = iSHTCUDA(nside_small, lmax=lmax, mmax=mmax)

        # Forward transform
        coeff = sht(signal)
        assert coeff.shape[:-2] == batch_shape, f"Expected batch shape {batch_shape}, got {coeff.shape[:-2]}"

        # Inverse transform
        signal_back = isht(coeff)
        assert signal_back.shape == signal.shape, f"Shape mismatch: {signal_back.shape} vs {signal.shape}"
