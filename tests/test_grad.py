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

nside = 32
npix = 12* nside**2
signal = torch.randn(npix, dtype = torch.float32).to(device)


quad_weights = 'ring'
lmax = 2*nside+1
mmax = lmax

sht = SHT(nside, lmax = lmax, mmax = mmax, quad_weights = quad_weights).to(device)
isht = iSHT(nside, lmax = lmax, mmax = mmax).to(device)

coeff = sht(signal)

sht_cuda = SHTCUDA(nside, lmax = lmax, mmax = mmax, quad_weights = quad_weights).to(device)
isht_cuda = iSHTCUDA(nside, lmax = lmax, mmax = mmax).to(device)

signal1 = torch.clone(signal)
signal2 = torch.clone(signal)

signal1.requires_grad_(True)
signal2.requires_grad_(True)

c1 = sht(signal1)
c2 = sht_cuda(signal2)

c1.backward(torch.clone(c1))
c2.backward(torch.clone(c1))

diff = signal1.grad - signal2.grad

print('-----------------------')
print('Mean of Autograd of SHT: ', torch.mean(signal1.grad.abs()))
print('Mean of manual of SHT: ', torch.mean(signal2.grad.abs()))
print('diff between the grad of SHT: ', torch.mean(diff.abs()))
print('ratio',  torch.mean(diff.abs())/torch.mean(signal1.grad.abs()))

coeff1 = torch.clone(coeff)
coeff2 = torch.clone(coeff)

coeff1.requires_grad_(True)
coeff2.requires_grad_(True)

s1 = isht(coeff1)
s2 = isht_cuda(coeff2)

s1.backward(torch.clone(s1))
s2.backward(torch.clone(s1))

diff = coeff1.grad - coeff2.grad

print('Mean of Autograd of iSHT: ', torch.mean(coeff1.grad.abs()))
print('Mean of manual of iSHT: ', torch.mean(coeff2.grad.abs()))
print('diff between the grad of iSHT: ', torch.mean(diff.abs()))
print('ratio',  torch.mean(diff.abs())/torch.mean(coeff1.grad.abs()))