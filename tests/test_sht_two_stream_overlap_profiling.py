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

from cuhpx import SHTCUDA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nside = int(input('nside: '))
nbatch = int(input('batch size: '))
npix = 12 * nside**2

signal = torch.randn(nbatch, npix, dtype=torch.float32).to(device)

quad_weights = 'ring'
lmax = int(input('lmax: '))
mmax = lmax

sht = SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights=quad_weights)

for _ in range(10):
    torch.cuda.nvtx.range_push("SHTCUDA batch")
    coeff = sht(signal)
    torch.cuda.nvtx.range_pop()
