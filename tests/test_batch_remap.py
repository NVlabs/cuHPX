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
import cuhpx

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nside = int(input('nside: '))
m = int(input('m: '))
n = int(input('n: '))

npix = 12* nside**2
signal = torch.randn((m,n,npix), dtype = torch.float32).to(device)

signal_dest = cuhpx.ring2nest(signal, nside)

signal_1by1 = torch.zeros((m,n,npix),  dtype = torch.float32).to(device)

for i in range(m):
	for j in range(n):
		signal_1by1[i,j,:] = cuhpx.ring2nest(signal[i,j,:], nside)


print("whether batch and one by one the same: ", torch.equal(signal_dest, signal_1by1))
