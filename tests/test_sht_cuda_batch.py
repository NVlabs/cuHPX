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

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nside = int(input('nside: '))
m = int(input('m, the first dim: '))
n = int(input('n, the second dim: '))

npix = 12* nside**2
signal = torch.randn(m,n, npix, dtype = torch.float32).to(device)

quad_weights = 'ring'
lmax = 2*nside+1
mmax = lmax

sht = SHTCUDA(nside, lmax = lmax, mmax = mmax, quad_weights = quad_weights)
isht = iSHTCUDA(nside, lmax = lmax, mmax = mmax)

coeff = sht(signal)
c = torch.zeros_like(coeff)

for i in range(m):
	for j in range(n):
		c[i,j,:] = sht(signal[i,j,:])

diff = (coeff - c).abs()
print('diff between batch and single, sht',torch.sqrt(torch.mean(diff.abs()**2)))

s1 = isht(coeff)
s2 = torch.zeros_like(s1)

for i in range(m):
	for j in range(n):
		s2[i,j,:] = isht(coeff[i,j,:])

diff = s1 - s2
print('diff between batch and single, isht',torch.sqrt(torch.mean(diff.abs()**2)))

