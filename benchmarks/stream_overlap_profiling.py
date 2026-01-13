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

"""
Benchmark: Profile CUDA stream overlap for SHT operations.

This script profiles the behavior of SHT operations across multiple CUDA streams
using NVTX markers for visualization in NSight Systems.

Usage:
    # Basic run
    python benchmarks/stream_overlap_profiling.py

    # With NSight Systems profiling
    nsys profile python benchmarks/stream_overlap_profiling.py
"""

import torch
import torch.cuda.nvtx as nvtx

from cuhpx import cuhpx_fft


def main():
    """Run stream overlap profiling benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Error: This benchmark requires CUDA")
        return

    # Configuration
    nside = int(input("nside: "))
    m = int(input("m, the first dim: "))
    n = int(input("n, the second dim: "))

    npix = 12 * nside**2
    lmax = 2 * nside + 1

    print("\nBenchmark configuration:")
    print(f"  nside: {nside}")
    print(f"  batch dims: ({m}, {n})")
    print(f"  npix: {npix}")
    print(f"  lmax: {lmax}")
    print()

    # Create input signals
    signal1 = torch.randn(m, n, npix, dtype=torch.float32, device=device)
    signal2 = signal1.clone()

    # Create two CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Profile with NVTX markers
    print("Running profiled stream operations...")

    # Perform SHT on stream1
    nvtx.range_push("Stream 1 SHT Operation")
    with torch.cuda.stream(stream1):
        result1 = cuhpx_fft.healpix_rfft_batch(signal1, nside, nside)
    stream1.synchronize()
    nvtx.range_pop()

    # Perform SHT on stream2
    nvtx.range_push("Stream 2 SHT Operation")
    with torch.cuda.stream(stream2):
        result2 = cuhpx_fft.healpix_rfft_batch(signal2, nside, nside)
    stream2.synchronize()
    nvtx.range_pop()

    # Compare results
    nvtx.range_push("Compare Results")
    comparison = torch.allclose(result1, result2)
    nvtx.range_pop()

    print(f"Results from two streams identical: {comparison}")
    print("Use 'nsys profile' to visualize stream overlap")


if __name__ == "__main__":
    main()
