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

"""Tests for CUDA stream support in SHT operations."""

import pytest
import torch

from cuhpx import cuhpx_fft


@pytest.mark.cuda
class TestCUDAStreams:
    """Test CUDA stream support for SHT operations."""

    @pytest.fixture
    def batch_signal(self, device, nside_small, dtype):
        """Create a batched signal for stream tests."""
        m, n = 2, 3
        npix = 12 * nside_small**2
        return torch.randn(m, n, npix, dtype=dtype, device=device)

    def test_different_streams_same_result(self, device, nside_small, batch_signal):
        """Test that operations on different streams produce identical results."""
        signal1 = batch_signal.clone()
        signal2 = batch_signal.clone()

        nside = nside_small

        # Create two CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        # Perform SHT on stream1
        with torch.cuda.stream(stream1):
            result1 = cuhpx_fft.healpix_rfft_batch(signal1, nside, nside)
        stream1.synchronize()

        # Perform SHT on stream2
        with torch.cuda.stream(stream2):
            result2 = cuhpx_fft.healpix_rfft_batch(signal2, nside, nside)
        stream2.synchronize()

        # Results should be identical
        assert torch.allclose(result1, result2), "Results from different streams should be identical"

    def test_stream_does_not_corrupt_data(self, device, nside_small, batch_signal):
        """Test that stream execution doesn't corrupt input data."""
        original = batch_signal.clone()
        signal = batch_signal.clone()

        nside = nside_small

        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            _ = cuhpx_fft.healpix_rfft_batch(signal, nside, nside)
        stream.synchronize()

        # Input should not be modified
        assert torch.allclose(signal, original), "Stream operation corrupted input data"

    def test_default_stream_works(self, device, nside_small, dtype):
        """Test that operations work on default stream."""
        npix = 12 * nside_small**2
        signal = torch.randn(npix, dtype=dtype, device=device)

        # Should work without explicit stream
        result = cuhpx_fft.healpix_rfft_batch(signal.unsqueeze(0), nside_small, nside_small)

        assert result is not None
        assert not torch.isnan(result).any(), "NaN in result"
