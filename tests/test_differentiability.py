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

"""Tests for differentiability of SHT operations (gradient-based optimization)."""

import pytest
import torch
import torch.nn as nn

from cuhpx import SHT, iSHTCUDA


def _generate_synthetic_signal(nside, lmax, mmax, device, seed=42, bandwidth_fraction=0.5):
    """Generate a synthetic bandlimited signal for testing.

    Args:
        nside: HEALPix nside parameter.
        lmax: Maximum degree l for the transform.
        mmax: Maximum order m for the transform.
        device: Torch device.
        seed: Random seed.
        bandwidth_fraction: Fraction of lmax to use for signal bandwidth (0 < frac <= 1).
            A value < 1 ensures the signal is strictly bandlimited below lmax.

    Returns:
        Signal tensor on device.
    """
    torch.manual_seed(seed)

    # Create bandlimited signal via inverse SHT
    from cuhpx import iSHT

    isht = iSHT(nside, lmax=lmax, mmax=mmax)

    # Only populate coefficients up to a fraction of lmax to ensure
    # the signal is strictly bandlimited below the transform's lmax
    max_l = int(lmax * bandwidth_fraction)
    max_l = max(1, min(max_l, lmax - 1))  # Ensure at least l=0 and at most lmax-1

    # Random coefficients with decay for higher l
    coeff = torch.zeros((lmax, mmax), dtype=torch.complex128)
    for l_idx in range(max_l):
        for m_idx in range(min(l_idx + 1, mmax)):
            # Decay amplitude with l for realistic signal
            amplitude = 1.0 / (1 + l_idx)
            coeff[l_idx, m_idx] = amplitude * (torch.randn(1) + 1j * torch.randn(1))

    signal = isht(coeff)
    return signal.to(device)


class SpectralModel(nn.Module):
    """Neural network module with learnable spherical harmonic coefficients."""

    def __init__(self, nside, lmax, mmax, device):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(lmax, mmax, dtype=torch.complex128))
        self.isht = iSHTCUDA(nside, lmax=lmax, mmax=mmax).to(device)

    def forward(self):
        return self.isht(self.coeffs)


@pytest.mark.cuda
@pytest.mark.slow
class TestDifferentiability:
    """Test differentiability through gradient-based optimization."""

    @pytest.fixture
    def nside_opt(self):
        """Return nside for optimization tests (smaller for speed)."""
        return 32

    @pytest.fixture
    def lmax_opt(self, nside_opt):
        """Return lmax for optimization tests."""
        return 2 * nside_opt + 1

    @pytest.fixture
    def target_signal(self, device, nside_opt, lmax_opt):
        """Create target signal for optimization."""
        return _generate_synthetic_signal(nside_opt, lmax_opt, lmax_opt, device)

    def test_spectral_model_optimization(self, device, nside_opt, lmax_opt, target_signal):
        """Test that spectral model can be optimized via gradient descent."""
        model = SpectralModel(nside_opt, lmax_opt, lmax_opt, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

        initial_loss = None
        final_loss = None

        # Run a few optimization steps
        n_iterations = 50
        for i in range(n_iterations):
            loss = (model() - target_signal).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1, (
            f"Optimization failed: initial loss = {initial_loss}, " f"final loss = {final_loss}"
        )

    def test_gradient_exists_and_valid(self, device, nside_opt, lmax_opt, target_signal):
        """Test that gradients exist and are valid (no NaN/Inf)."""
        model = SpectralModel(nside_opt, lmax_opt, lmax_opt, device).to(device)

        loss = (model() - target_signal).pow(2).mean()
        loss.backward()

        assert model.coeffs.grad is not None, "Gradient not computed"
        assert not torch.isnan(model.coeffs.grad).any(), "NaN in gradients"
        assert not torch.isinf(model.coeffs.grad).any(), "Inf in gradients"

    def test_sht_forward_differentiable(self, device, nside_opt, lmax_opt):
        """Test that SHT forward pass is differentiable."""
        npix = 12 * nside_opt**2
        signal = torch.randn(npix, dtype=torch.float32, device=device, requires_grad=True)

        sht = SHT(nside_opt, lmax=lmax_opt, mmax=lmax_opt, quad_weights="ring").to(device)

        coeff = sht(signal)
        loss = coeff.abs().sum()
        loss.backward()

        assert signal.grad is not None, "SHT forward not differentiable"
        assert signal.grad.shape == signal.shape, "Gradient shape mismatch"

    def test_isht_forward_differentiable(self, device, nside_opt, lmax_opt):
        """Test that iSHT forward pass is differentiable."""
        coeff = torch.randn(lmax_opt, lmax_opt, dtype=torch.complex128, device=device, requires_grad=True)

        isht = iSHTCUDA(nside_opt, lmax=lmax_opt, mmax=lmax_opt).to(device)

        signal = isht(coeff)
        loss = signal.abs().sum()
        loss.backward()

        assert coeff.grad is not None, "iSHT forward not differentiable"
        assert coeff.grad.shape == coeff.shape, "Gradient shape mismatch"
