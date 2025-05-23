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

import torch
from cuhpx import SHTCUDA, iSHTCUDA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nside = int(input('nside: '))
lmax = int(input('lmax: '))
nbatch = int(input('nbatch: '))

npix = 12 * nside**2
signal = torch.randn(nbatch, npix, dtype=torch.float32).to(device)
quad_weights = 'ring'
mmax = lmax

sht = SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights=quad_weights)
isht = iSHTCUDA(nside, lmax=lmax, mmax=mmax)

for _ in range(2):
    coeff = sht(signal)
    signal_round_trip = isht(coeff)

# Begin NVTX range for SHT
torch.cuda.nvtx.range_push("SHTCUDA batch")

coeff = sht(signal)

# End NVTX range for SHT
torch.cuda.nvtx.range_pop()

# Begin NVTX range for iSHT
torch.cuda.nvtx.range_push("iSHTCUDA batch")

signal_round_trip = isht(coeff)

# End NVTX range for iSHT
torch.cuda.nvtx.range_pop()

for _ in range(5):
    coeff = sht(signal)
    signal_round_trip = isht(coeff)
