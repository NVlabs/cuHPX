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

"""Pytest configuration and shared fixtures for cuHPX tests."""

import pytest
import torch

# Check if CUDA is available once at module load
CUDA_AVAILABLE = torch.cuda.is_available()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: marks tests as requiring CUDA GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


@pytest.fixture
def device():
    """Provide CUDA device, skip test if unavailable."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(params=[32, 64])
def nside(request):
    """Parametrize nside values for tests."""
    return request.param


@pytest.fixture
def nside_small():
    """Return a small nside value for quick tests."""
    return 32


@pytest.fixture
def lmax(nside):
    """Return lmax based on nside (healpy default: 3*nside - 1)."""
    return 3 * nside - 1


@pytest.fixture
def mmax(lmax):
    """Return mmax (same as lmax by default)."""
    return lmax


@pytest.fixture
def npix(nside):
    """Return number of HEALPix pixels for given nside."""
    return 12 * nside**2


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    """Parametrize float dtypes for tests."""
    return request.param


@pytest.fixture
def complex_dtype(dtype):
    """Return corresponding complex dtype for a real dtype."""
    return torch.complex64 if dtype == torch.float32 else torch.complex128


# Tolerance configurations for different test types
# These are based on empirical measurements of algorithm accuracy

# For CUDA vs PyTorch implementation comparison (same algorithm, different impl)
IMPL_COMPARISON_TOL = {
    torch.float32: {"sht": (1e-4, 2e-5), "isht": (1e-3, 1e-1)},
    torch.float64: {"sht": (1e-8, 1e-8), "isht": (1e-5, 1e-5)},
}

# For Bluestein vs standard comparison (different algorithms, both float64 internally)
# The Bluestein algorithm has inherent numerical differences from standard FFT
# that are algorithm-limited (~1e-6 for SHT, ~1e-4 for iSHT) not precision-limited
BLUESTEIN_COMPARISON_TOL = {
    "sht": (1e-5, 1e-5),
    "isht": (1e-4, 1e-4),
}

# For roundtrip tests (algorithm-limited, not precision-limited)
# These tolerances are the same for both dtypes because error is algorithmic
ROUNDTRIP_TOL = {"rtol": 0.01, "atol": 0.05}


def get_impl_tol(dtype, transform_type):
    """Get (rtol, atol) for implementation comparison tests.

    Args:
        dtype: torch.float32 or torch.float64
        transform_type: "sht" or "isht"

    Returns:
        Tuple of (rtol, atol)
    """
    return IMPL_COMPARISON_TOL[dtype][transform_type]


def get_roundtrip_tol():
    """Get (rtol, atol) for roundtrip tests.

    Roundtrip error is algorithm-limited, not precision-limited,
    so same tolerances for float32 and float64.
    """
    return ROUNDTRIP_TOL["rtol"], ROUNDTRIP_TOL["atol"]


def get_bluestein_tol(transform_type):
    """Get (rtol, atol) for Bluestein vs standard comparison tests.

    Bluestein uses a different FFT algorithm (convolution-based) which
    has inherent numerical differences from standard FFT. These are
    algorithm-limited, not precision-limited.

    Args:
        transform_type: "sht" or "isht"

    Returns:
        Tuple of (rtol, atol)
    """
    return BLUESTEIN_COMPARISON_TOL[transform_type]
