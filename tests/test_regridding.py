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

"""Tests for grid regridding operations."""

import random

import pytest
import torch

import cuhpx
from cuhpx import Grid, Regridding


def _generate_bandlimited_coeffs(
    lmax, mmax, n_nonzero=100, seed=42, bandwidth_fraction=0.5, complex_dtype=torch.complex128
):
    """Generate sparse spherical harmonic coefficients for testing.

    Args:
        lmax: Maximum degree l (size of coefficient array).
        mmax: Maximum order m (size of coefficient array).
        n_nonzero: Number of non-zero coefficients.
        seed: Random seed for reproducibility.
        bandwidth_fraction: Fraction of lmax/mmax to populate (default 0.5).
            Using a fraction < 1.0 ensures the signal is strictly bandlimited
            with headroom for perfect roundtrip through SHT.
        complex_dtype: Complex dtype for coefficients (default torch.complex128).

    Returns:
        Complex tensor of shape (lmax, mmax) with sparse coefficients.
    """
    random.seed(seed)
    coeff = torch.zeros((lmax, mmax), dtype=complex_dtype)

    # Limit the bandwidth to ensure strict bandlimiting
    max_l = max(1, int(lmax * bandwidth_fraction))
    max_m = max(1, int(mmax * bandwidth_fraction))

    for _ in range(n_nonzero):
        v = random.random()
        x = random.randint(0, max_l - 1)
        y = random.randint(0, min(x, max_m - 1))
        coeff[x, y] = v

    return coeff


@pytest.mark.cuda
class TestRegridding:
    """Test regridding between different grid types."""

    def test_healpix_equiangular_roundtrip(self, device, nside, dtype, complex_dtype):
        """Test HEALPix -> Equiangular -> HEALPix roundtrip."""
        # Use lmax that is compatible with both grids
        # For equiangular grid of size (2*nside, 4*nside), lmax should be < 2*nside
        lmax = 2 * nside - 1
        mmax = lmax

        src_grid = Grid("healpix", nside)
        dest_grid = Grid("equiangular", (2 * nside, 4 * nside))

        # Create bandlimited signal
        isht = cuhpx.iSHT(nside, lmax=lmax, mmax=mmax)
        coeff = _generate_bandlimited_coeffs(lmax, mmax, n_nonzero=50, complex_dtype=complex_dtype)
        signal = isht(coeff).to(device).to(dtype)

        hpx2eq = Regridding(src_grid, dest_grid, lmax=lmax, mmax=mmax, device=device)
        eq2hpx = Regridding(dest_grid, src_grid, lmax=lmax, mmax=mmax, device=device)

        signal_eq = hpx2eq.execute(signal)
        signal_back = eq2hpx.execute(signal_eq)

        # For bandlimited signals, roundtrip should be accurate
        # rtol=0.02 (2% relative tolerance), atol=0.05 for regridding which has more numerical error
        assert torch.allclose(
            signal_back, signal, rtol=0.02, atol=0.05
        ), f"Regridding roundtrip failed: max diff = {(signal_back - signal).abs().max():.2e}"

    def test_regridding_preserves_dtype(self, device, nside_small, dtype, complex_dtype):
        """Test that regridding preserves data type."""
        nside = nside_small
        lmax = 2 * nside - 1
        mmax = lmax

        isht = cuhpx.iSHT(nside, lmax=lmax, mmax=mmax)
        coeff = _generate_bandlimited_coeffs(lmax, mmax, n_nonzero=50, complex_dtype=complex_dtype)
        signal = isht(coeff).to(device).to(dtype)

        src_grid = Grid("healpix", nside)
        dest_grid = Grid("equiangular", (2 * nside, 4 * nside))
        regrid = Regridding(src_grid, dest_grid, lmax=lmax, mmax=mmax, device=device)

        output = regrid.execute(signal)
        assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"

    def test_grid_creation(self):
        """Test Grid class instantiation."""
        # HEALPix grid
        hpx_grid = Grid("healpix", 64)
        assert hpx_grid.grid == "healpix"
        assert hpx_grid.nside == 64

        # Equiangular grid
        eq_grid = Grid("equiangular", (128, 256))
        assert eq_grid.grid == "equiangular"
        assert eq_grid.nlat == 128
        assert eq_grid.nlon == 256

    def test_regridding_output_shape(self, device, nside_small, dtype):
        """Test that regridding produces correct output shapes."""
        nside = nside_small
        lmax = 2 * nside - 1
        mmax = lmax

        npix = 12 * nside**2
        signal = torch.randn(npix, dtype=dtype, device=device)

        src_grid = Grid("healpix", nside)
        nlat, nlon = 2 * nside, 4 * nside
        dest_grid = Grid("equiangular", (nlat, nlon))

        regrid = Regridding(src_grid, dest_grid, lmax=lmax, mmax=mmax, device=device)
        output = regrid.execute(signal)

        expected_shape = (nlat, nlon)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    def test_healpix_to_equiangular(self, device, nside_small, dtype):
        """Test one-way regridding from HEALPix to equiangular grid."""
        nside = nside_small
        lmax = 2 * nside - 1
        mmax = lmax

        npix = 12 * nside**2
        torch.manual_seed(42)
        signal = torch.randn(npix, dtype=dtype, device=device)

        src_grid = Grid("healpix", nside)
        dest_grid = Grid("equiangular", (2 * nside, 4 * nside))

        regrid = Regridding(src_grid, dest_grid, lmax=lmax, mmax=mmax, device=device)
        output = regrid.execute(signal)

        # Basic sanity checks
        assert not torch.isnan(output).any(), "NaN in regridding output"
        assert not torch.isinf(output).any(), "Inf in regridding output"
        assert output.shape == (2 * nside, 4 * nside)
