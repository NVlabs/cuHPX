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

"""Tests for gradient computation of SHT/iSHT operations."""

import pytest
import torch

from cuhpx import SHT, SHTCUDA, iSHT, iSHTCUDA


@pytest.mark.cuda
class TestSHTGradients:
    """Test gradient computation for spherical harmonic transforms."""

    @pytest.fixture
    def nside_fixed(self):
        """Return fixed nside for gradient tests."""
        return 32

    @pytest.fixture
    def lmax_fixed(self, nside_fixed):
        """Return lmax for fixed nside."""
        return 2 * nside_fixed + 1

    @pytest.fixture
    def signal(self, device, nside_fixed, dtype):
        """Create a random test signal."""
        npix = 12 * nside_fixed**2
        return torch.randn(npix, dtype=dtype, device=device)

    @pytest.fixture
    def sht_autograd(self, device, nside_fixed, lmax_fixed):
        """Create SHT with autograd support."""
        return SHT(nside_fixed, lmax=lmax_fixed, mmax=lmax_fixed, quad_weights="ring").to(device)

    @pytest.fixture
    def isht_autograd(self, device, nside_fixed, lmax_fixed):
        """Create iSHT with autograd support."""
        return iSHT(nside_fixed, lmax=lmax_fixed, mmax=lmax_fixed).to(device)

    @pytest.fixture
    def sht_cuda(self, device, nside_fixed, lmax_fixed):
        """Create SHTCUDA with custom backward."""
        return SHTCUDA(nside_fixed, lmax=lmax_fixed, mmax=lmax_fixed, quad_weights="ring").to(device)

    @pytest.fixture
    def isht_cuda(self, device, nside_fixed, lmax_fixed):
        """Create iSHTCUDA with custom backward."""
        return iSHTCUDA(nside_fixed, lmax=lmax_fixed, mmax=lmax_fixed).to(device)

    def test_sht_gradient_flow(self, signal, sht_cuda):
        """Test that gradients flow through SHTCUDA."""
        signal_grad = signal.clone().requires_grad_(True)
        coeff = sht_cuda(signal_grad)

        # Create a scalar loss
        loss = coeff.abs().sum()
        loss.backward()

        assert signal_grad.grad is not None, "Gradient not computed for SHT"
        assert not torch.isnan(signal_grad.grad).any(), "NaN in SHT gradient"
        assert not torch.isinf(signal_grad.grad).any(), "Inf in SHT gradient"

    def test_isht_gradient_flow(self, signal, sht_autograd, isht_cuda):
        """Test that gradients flow through iSHTCUDA."""
        coeff = sht_autograd(signal)
        coeff_grad = coeff.clone().requires_grad_(True)

        signal_out = isht_cuda(coeff_grad)

        # Create a scalar loss
        loss = signal_out.abs().sum()
        loss.backward()

        assert coeff_grad.grad is not None, "Gradient not computed for iSHT"
        assert not torch.isnan(coeff_grad.grad).any(), "NaN in iSHT gradient"
        assert not torch.isinf(coeff_grad.grad).any(), "Inf in iSHT gradient"
