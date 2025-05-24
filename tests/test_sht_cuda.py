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

from cuhpx import SHT, SHTCUDA, iSHT, iSHTCUDA

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nside = int(input('nside: '))
lmax = int(input('lmax: '))
npix = 12 * nside**2
signal = torch.randn(npix, dtype=torch.float32).to(device)

quad_weights = 'ring'

mmax = lmax

sht = SHT(nside, lmax=lmax, mmax=mmax, quad_weights=quad_weights).to(device)
isht = iSHT(nside, lmax=lmax, mmax=mmax).to(device)

coeff = sht(signal)

sht_cuda = SHTCUDA(nside, lmax=lmax, mmax=mmax, quad_weights=quad_weights)
isht_cuda = iSHTCUDA(nside, lmax=lmax, mmax=mmax)

diff = sht(signal) - sht_cuda(signal)
print('diff between pytorch and cuda, sht', torch.sqrt(torch.mean(diff.abs() ** 2)))

diff = isht(torch.clone(coeff)) - isht_cuda(torch.clone(coeff))
print('diff between pytorch and cuda, isht', torch.sqrt(torch.mean(diff.abs() ** 2)))
