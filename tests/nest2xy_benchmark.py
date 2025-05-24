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

import time

import healpy as hp
import numpy as np
import torch
from earth2grid.healpix import HEALPIX_PAD_XY, PixelOrder

import cuhpx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nside = int(input('nside: '))
nbatch = int(input('batch size: '))

npix = 12 * nside**2

data = torch.rand(npix, dtype=torch.float)
data_np = data.numpy()
data = data.to(device)

# Prepare lists to collect times
cuhpx_times = []
hp_times = []

for _ in range(15):
    # Measure cuhpx.nest2flat
    start_cuhpx = torch.cuda.Event(enable_timing=True)
    end_cuhpx = torch.cuda.Event(enable_timing=True)
    start_cuhpx.record()

    data = cuhpx.nest2flat(data, "N", True, nside)

    end_cuhpx.record()
    torch.cuda.synchronize()  # Ensure timing has finished
    cuhpx_times.append(start_cuhpx.elapsed_time(end_cuhpx))

for _ in range(15):
    # Measure hp.reorder
    start_hp = time.time()
    hp.reorder(data_np, n2r=True)
    end_hp = time.time()
    hp_times.append((end_hp - start_hp) * 1000)  # Convert to milliseconds

# Discard the first 5 timings and calculate the average of the next 10
avg_cuhpx_time = np.mean(cuhpx_times[5:])
avg_hp_time = np.mean(hp_times[5:])

print(f'Average time for non-batch cuhpx.nest2flat: {avg_cuhpx_time} ms')
print(f'Average time for healpy reorder: {avg_hp_time} ms')

data_batch = torch.rand(nbatch, npix, dtype=torch.float).to(device)
# Measure cuhpx.nest2flat
start_batch = torch.cuda.Event(enable_timing=True)
end_batch = torch.cuda.Event(enable_timing=True)
for i in range(15):
    if i == 5:
        start_batch.record()

    data_batch = cuhpx.nest2flat(data_batch, "N", True, nside)

end_batch.record()
end_batch.synchronize()  # Ensure timing has finished
avg_batch_time = start_batch.elapsed_time(end_batch) / 10 / nbatch

print(f'Average time for batch cuhpx.nest2flat: {avg_batch_time} ms')
print(f'    Speed up using non-batch cuhpx over healpy: {avg_hp_time/avg_cuhpx_time}')
print(f'    Speed up using batch cuhpx over healpy: {avg_hp_time/avg_batch_time}')

nest = PixelOrder.NEST
reorder = torch.compile(PixelOrder.NEST.to_xy_cuda)
# Measure cuhpx.nest2flat
start_batch = torch.cuda.Event(enable_timing=True)
end_batch = torch.cuda.Event(enable_timing=True)
for i in range(15):
    if i == 5:
        start_batch.record()

    data_batch = reorder(data_batch, HEALPIX_PAD_XY)

end_batch.record()
end_batch.synchronize()  # Ensure timing has finished
avg_e2grid_time = start_batch.elapsed_time(end_batch) / 10 / nbatch

print(f'Average time for batch PixelOrder.NEST.to_xy_cuda: {avg_e2grid_time} ms')
print(f'    Speed up using non-batch cuhpx over e2grid: {avg_e2grid_time/avg_cuhpx_time}')
print(f'    Speed up using batch cuhpx over e2grid: {avg_e2grid_time/avg_batch_time}')
