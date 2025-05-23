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

import random
import torch
from cuhpx import SHT, iSHT
from cuhpx import SHTCUDA, iSHTCUDA
import cuhpx_fft
import torch.cuda.nvtx as nvtx

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# User input
nside = int(input('nside: '))
m = int(input('m, the first dim: '))
n = int(input('n, the second dim: '))

npix = 12 * nside**2
signal1 = torch.randn(m, n, npix, dtype=torch.float32).to(device)
signal2 = torch.clone(signal1)

quad_weights = 'ring'
lmax = 2 * nside + 1
mmax = lmax

# Create two CUDA streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

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

# Synchronize streams and compare results
nvtx.range_push("Compare Results")
comparison = torch.allclose(result1, result2)
print("Are the results from the two streams identical?", comparison)
nvtx.range_pop()
