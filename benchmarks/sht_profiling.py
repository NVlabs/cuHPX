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
Benchmark: Profile SHTCUDA batched operations with NVTX markers.

This script profiles the performance of batched spherical harmonic transforms
using NVTX markers for visualization in NSight Systems or NSight Compute.

Usage:
    # Basic run
    python benchmarks/sht_profiling.py

    # With NSight Systems profiling
    nsys profile python benchmarks/sht_profiling.py
"""

import torch

from cuhpx import SHTCUDA


def main():
    """Run SHT profiling benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Error: This benchmark requires CUDA")
        return

    # Configuration
    nside = int(input("nside: "))
    nbatch = int(input("batch size: "))
    lmax = int(input("lmax: "))
    mmax = lmax
    npix = 12 * nside**2

    print("\nBenchmark configuration:")
    print(f"  nside: {nside}")
    print(f"  batch_size: {nbatch}")
    print(f"  lmax: {lmax}")
    print(f"  npix: {npix}")
    print()

    # Create input signal
    signal = torch.randn(nbatch, npix, dtype=torch.float32, device=device)

    # Create SHT transform
    sht = SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights="ring")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = sht(signal)
    torch.cuda.synchronize()

    # Profile with NVTX markers
    print("Running profiled iterations...")
    n_iterations = 10

    for i in range(n_iterations):
        torch.cuda.nvtx.range_push(f"SHTCUDA batch iter {i}")
        _ = sht(signal)
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    print(f"Completed {n_iterations} iterations")
    print("Use 'nsys profile' to capture detailed profiling data")


if __name__ == "__main__":
    main()
