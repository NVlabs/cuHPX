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

"""Tests for CUDA graph compatibility of SHTCUDA/iSHTCUDA."""

import pytest
import torch

from cuhpx import SHTCUDA, iSHTCUDA


@pytest.mark.cuda
def test_shtcuda_cuda_graph_capture(device, nside_small, dtype):
    """Test that SHTCUDA can be captured in a CUDA graph."""
    lmax = 3 * nside_small - 1
    mmax = lmax
    npix = 12 * nside_small**2

    sht = SHTCUDA(nside_small, lmax=lmax, mmax=mmax)

    # Warmup (triggers lazy initialization)
    signal = torch.randn(npix, dtype=dtype, device=device)
    _ = sht(signal)

    # Prepare input for graph capture
    static_input = torch.randn(npix, dtype=dtype, device=device)

    # Warmup the specific input size
    _ = sht(static_input.clone())
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()

    # Capture graph - use the default stream, not a separate one
    with torch.cuda.graph(g):
        static_output = sht(static_input)

    torch.cuda.synchronize()

    # Prepare test input and compute expected result BEFORE replay
    new_input = torch.randn(npix, dtype=dtype, device=device)

    # Copy to static input and replay
    static_input.copy_(new_input)
    g.replay()
    torch.cuda.synchronize()

    # Copy result before computing expected (to avoid interference)
    graph_result = static_output.clone()

    # Compute expected using direct call
    expected = sht(new_input)

    rtol = 1e-4 if dtype == torch.float32 else 1e-8
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(
        graph_result, expected, rtol=rtol, atol=atol
    ), f"CUDA graph output differs from direct call: max diff = {(graph_result - expected).abs().max():.2e}"


@pytest.mark.cuda
def test_ishtcuda_cuda_graph_capture(device, nside_small, dtype, complex_dtype):
    """Test that iSHTCUDA can be captured in a CUDA graph."""
    lmax = 3 * nside_small - 1
    mmax = lmax

    isht = iSHTCUDA(nside_small, lmax=lmax, mmax=mmax)

    # Warmup
    coeffs = torch.randn(lmax, mmax, dtype=complex_dtype, device=device)
    _ = isht(coeffs.clone())
    torch.cuda.synchronize()

    # Capture graph
    static_input = torch.randn(lmax, mmax, dtype=complex_dtype, device=device)

    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        static_output = isht(static_input)

    torch.cuda.synchronize()

    # Replay with new data
    new_input = torch.randn(lmax, mmax, dtype=complex_dtype, device=device)
    static_input.copy_(new_input)
    g.replay()
    torch.cuda.synchronize()

    # Copy result before computing expected
    graph_result = static_output.clone()

    # Verify correctness
    expected = isht(new_input)
    rtol = 1e-3 if dtype == torch.float32 else 1e-5
    atol = 1e-2 if dtype == torch.float32 else 1e-5
    assert torch.allclose(
        graph_result, expected, rtol=rtol, atol=atol
    ), f"CUDA graph output differs from direct call: max diff = {(graph_result - expected).abs().max():.2e}"


@pytest.mark.cuda
def test_shtcuda_batch_cuda_graph(device, nside_small, dtype):
    """Test SHTCUDA with batched input in CUDA graph."""
    lmax = 3 * nside_small - 1
    mmax = lmax
    npix = 12 * nside_small**2
    batch_size = 4

    sht = SHTCUDA(nside_small, lmax=lmax, mmax=mmax)

    # Warmup with batch
    signal = torch.randn(batch_size, npix, dtype=dtype, device=device)
    _ = sht(signal)
    torch.cuda.synchronize()

    # Capture graph
    static_input = torch.randn(batch_size, npix, dtype=dtype, device=device)

    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        static_output = sht(static_input)

    torch.cuda.synchronize()

    # Replay
    new_input = torch.randn(batch_size, npix, dtype=dtype, device=device)
    static_input.copy_(new_input)
    g.replay()
    torch.cuda.synchronize()

    # Copy result before computing expected
    graph_result = static_output.clone()

    expected = sht(new_input)
    rtol = 1e-4 if dtype == torch.float32 else 1e-8
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(
        graph_result, expected, rtol=rtol, atol=atol
    ), f"CUDA graph batch output differs: max diff = {(graph_result - expected).abs().max():.2e}"


@pytest.mark.cuda
def test_ishtcuda_batch_cuda_graph(device, nside_small, dtype, complex_dtype):
    """Test iSHTCUDA with batched input in CUDA graph."""
    lmax = 3 * nside_small - 1
    mmax = lmax
    batch_size = 4

    isht = iSHTCUDA(nside_small, lmax=lmax, mmax=mmax)

    # Warmup with batch
    coeffs = torch.randn(batch_size, lmax, mmax, dtype=complex_dtype, device=device)
    _ = isht(coeffs.clone())
    torch.cuda.synchronize()

    # Capture graph
    static_input = torch.randn(batch_size, lmax, mmax, dtype=complex_dtype, device=device)

    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        static_output = isht(static_input)

    torch.cuda.synchronize()

    # Replay
    new_input = torch.randn(batch_size, lmax, mmax, dtype=complex_dtype, device=device)
    static_input.copy_(new_input)
    g.replay()
    torch.cuda.synchronize()

    # Copy result before computing expected
    graph_result = static_output.clone()

    expected = isht(new_input)
    rtol = 1e-3 if dtype == torch.float32 else 1e-5
    atol = 1e-2 if dtype == torch.float32 else 1e-5
    assert torch.allclose(
        graph_result, expected, rtol=rtol, atol=atol
    ), f"CUDA graph batch output differs: max diff = {(graph_result - expected).abs().max():.2e}"


@pytest.mark.cuda
def test_roundtrip_cuda_graph(device, nside_small, dtype):
    """Test SHT+iSHT roundtrip in CUDA graph."""
    lmax = 3 * nside_small - 1
    mmax = lmax
    npix = 12 * nside_small**2

    sht = SHTCUDA(nside_small, lmax=lmax, mmax=mmax)
    isht = iSHTCUDA(nside_small, lmax=lmax, mmax=mmax)

    # Warmup
    signal = torch.randn(npix, dtype=dtype, device=device)
    _ = isht(sht(signal))
    torch.cuda.synchronize()

    # Capture roundtrip
    static_input = torch.randn(npix, dtype=dtype, device=device)

    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        coeffs = sht(static_input)
        static_output = isht(coeffs)

    torch.cuda.synchronize()

    # Replay
    new_input = torch.randn(npix, dtype=dtype, device=device)
    static_input.copy_(new_input)
    g.replay()
    torch.cuda.synchronize()

    # Copy result before computing expected
    graph_result = static_output.clone()

    expected = isht(sht(new_input))
    rtol = 1e-3 if dtype == torch.float32 else 1e-5
    atol = 1e-2 if dtype == torch.float32 else 1e-5
    assert torch.allclose(
        graph_result, expected, rtol=rtol, atol=atol
    ), f"CUDA graph roundtrip differs: max diff = {(graph_result - expected).abs().max():.2e}"


@pytest.mark.cuda
def test_cuda_graph_multiple_replays(device, nside_small, dtype):
    """Test that CUDA graph can be replayed multiple times correctly."""
    lmax = 3 * nside_small - 1
    mmax = lmax
    npix = 12 * nside_small**2

    sht = SHTCUDA(nside_small, lmax=lmax, mmax=mmax)

    # Warmup
    _ = sht(torch.randn(npix, dtype=dtype, device=device))
    torch.cuda.synchronize()

    # Capture
    static_input = torch.randn(npix, dtype=dtype, device=device)

    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        static_output = sht(static_input)

    torch.cuda.synchronize()

    # Multiple replays
    rtol = 1e-4 if dtype == torch.float32 else 1e-8
    atol = 1e-5 if dtype == torch.float32 else 1e-10

    for i in range(5):
        new_input = torch.randn(npix, dtype=dtype, device=device)
        static_input.copy_(new_input)
        g.replay()
        torch.cuda.synchronize()

        # Copy result before computing expected
        graph_result = static_output.clone()

        expected = sht(new_input)
        assert torch.allclose(
            graph_result, expected, rtol=rtol, atol=atol
        ), f"CUDA graph replay {i} failed: max diff = {(graph_result - expected).abs().max():.2e}"
