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
Example: Spectral optimization on WMAP data using differentiable SHT.

This example demonstrates how to use cuHPX's differentiable inverse spherical
harmonic transform to fit spherical harmonic coefficients to WMAP sky map data.

Usage:
    python examples/wmap_optimization.py

Requirements:
    - healpy (pip install healpy)
    - CUDA-capable GPU
    - Network access to download WMAP data from NASA
"""

import os
import urllib.request

import healpy as hp
import torch
import torch.nn as nn

from cuhpx import iSHTCUDA


def download_wmap_data(filename="wmap_band_iqumap_r9_7yr_W_v4.fits"):
    """Download WMAP 7-year W-band sky map if not present."""
    if os.path.exists(filename):
        print(f"Using existing file: {filename}")
        return filename

    url = f"http://lambda.gsfc.nasa.gov/data/map/dr4/skymaps/7yr/raw/{filename}"
    print(f"Downloading WMAP data from {url}...")

    try:
        urllib.request.urlretrieve(url, filename)  # noqa: S310
        print(f"Downloaded: {filename}")
        return filename
    except Exception as e:
        raise RuntimeError(f"Failed to download WMAP data: {e}")


class SpectralModel(nn.Module):
    """Neural network module with learnable spherical harmonic coefficients.

    This model learns to represent a HEALPix map as spherical harmonic
    coefficients. The forward pass applies the inverse SHT to produce
    a pixel-space representation.
    """

    def __init__(self, nside, lmax, mmax, device):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(lmax, mmax, dtype=torch.complex128))
        self.isht = iSHTCUDA(nside, lmax=lmax, mmax=mmax).to(device)

    def forward(self):
        return self.isht(self.coeffs)


def main():
    """Run WMAP spectral optimization example."""
    # Configuration
    nside = int(input("Enter the nside value (e.g., 64, 128): "))
    lmax = int(input("Enter the lmax value (e.g., 2*nside+1): "))
    mmax = lmax
    n_iterations = 500
    learning_rate = 5e-2

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, running on CPU (will be slow)")

    # Download and load WMAP data
    fits_file = download_wmap_data()
    wmap_map_I = hp.read_map(fits_file)
    wmap = hp.ud_grade(wmap_map_I, nside)

    # Convert to torch tensor
    signal = torch.from_numpy(wmap).to(device)
    print(f"Loaded WMAP data: nside={nside}, npix={len(wmap)}")

    # Initialize model and optimizer
    model = SpectralModel(nside, lmax, mmax, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nOptimizing spherical harmonic coefficients (lmax={lmax})...")
    print("-" * 50)

    losses = []
    for iteration in range(n_iterations):
        # Forward pass
        prediction = model()
        loss = (prediction - signal).pow(2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iteration % 50 == 0 or iteration == n_iterations - 1:
            print(f"Iteration {iteration:4d}: MSE loss = {loss.item():.6e}")

    # Report results
    print("-" * 50)
    print(f"Final MSE loss: {losses[-1]:.6e}")
    print(f"Loss reduction: {losses[0] / losses[-1]:.1f}x")
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
