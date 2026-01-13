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

"""Tests comparing CUDA-accelerated SHT/iSHT against PyTorch reference implementation."""

import pytest
import torch
from conftest import get_impl_tol, get_roundtrip_tol

from cuhpx import SHT, SHTCUDA, iSHT, iSHTCUDA


@pytest.mark.cuda
def test_shtcuda_matches_sht(device, nside, lmax, mmax, dtype):
    """Test that SHTCUDA produces same results as PyTorch SHT."""
    npix = 12 * nside**2
    torch.manual_seed(42)
    signal = torch.randn(npix, dtype=dtype, device=device)

    sht = SHT(nside, lmax=lmax, mmax=mmax, quad_weights="ring").to(device)
    sht_cuda = SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights="ring")

    result_torch = sht(signal)
    result_cuda = sht_cuda(signal)

    rtol, atol = get_impl_tol(dtype, "sht")
    assert torch.allclose(
        result_torch, result_cuda, rtol=rtol, atol=atol
    ), f"SHTCUDA differs from SHT: max diff = {(result_torch - result_cuda).abs().max():.2e}"


@pytest.mark.cuda
def test_ishtcuda_matches_isht(device, nside, lmax, mmax, dtype, complex_dtype):
    """Test that iSHTCUDA produces same results as PyTorch iSHT."""
    torch.manual_seed(42)
    coeffs = torch.randn(lmax, mmax, dtype=complex_dtype, device=device)

    isht = iSHT(nside, lmax=lmax, mmax=mmax).to(device)
    isht_cuda = iSHTCUDA(nside, lmax=lmax, mmax=mmax)

    result_torch = isht(coeffs.clone())
    result_cuda = isht_cuda(coeffs.clone())

    rtol, atol = get_impl_tol(dtype, "isht")
    assert torch.allclose(
        result_torch, result_cuda, rtol=rtol, atol=atol
    ), f"iSHTCUDA differs from iSHT: max diff = {(result_torch - result_cuda).abs().max():.2e}"


@pytest.mark.cuda
def test_shtcuda_isht_cuda_roundtrip(device, nside, lmax, mmax, dtype, complex_dtype):
    """Test SHTCUDA + iSHTCUDA round-trip on bandlimited signals."""
    sht_cuda = SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights="ring")
    isht_cuda = iSHTCUDA(nside, lmax=lmax, mmax=mmax)

    # Create strictly band-limited signal
    torch.manual_seed(42)
    coeffs = torch.zeros((lmax, mmax), dtype=complex_dtype, device=device)
    max_l = min(lmax // 2, nside // 2)  # Use half of lmax for safety margin
    for l_idx in range(max_l):
        for m_idx in range(min(l_idx + 1, mmax)):
            coeffs[l_idx, m_idx] = torch.randn(1, dtype=dtype).item()

    signal = isht_cuda(coeffs)
    reconstructed = isht_cuda(sht_cuda(signal))

    # Roundtrip error is algorithm-limited, not precision-limited
    rtol, atol = get_roundtrip_tol()
    assert torch.allclose(
        reconstructed, signal, rtol=rtol, atol=atol
    ), f"CUDA roundtrip failed: max diff = {(reconstructed - signal).abs().max():.2e}"


@pytest.mark.cuda
def test_shtcuda_output_consistency(device, nside, lmax, mmax, dtype):
    """Test that SHTCUDA produces consistent output across multiple calls."""
    npix = 12 * nside**2
    torch.manual_seed(42)
    signal = torch.randn(npix, dtype=dtype, device=device)

    sht_cuda = SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights="ring")

    result1 = sht_cuda(signal.clone())
    result2 = sht_cuda(signal.clone())

    # Results should be exactly identical for deterministic operations
    assert torch.allclose(result1, result2, rtol=0, atol=0), "SHTCUDA produces inconsistent results"
