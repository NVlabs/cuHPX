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
import numpy as np
from cuhpx import SHTCUDA
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nside = int(input('nside: '))
lmax = int(input('lmax: '))
mmax= lmax
npix = 12 * nside**2

nbatch = int(input('nbatch: '))

data_torch = torch.rand(npix, dtype = torch.float)
data_batch = torch.rand(nbatch, npix, dtype = torch.float)

data_torch = data_torch.to(device)
data_batch = data_batch.to(device)

sht = SHTCUDA(nside, lmax=lmax, mmax=mmax).to(device)

_ = sht(data_torch)
_ = sht(data_batch)

# Timing SHTCUDA on single data
torch.cuda.synchronize(device)
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)
times_sht = []
for _ in range(15):
    start_time.record()
    _ = sht(data_torch)
    end_time.record()
    torch.cuda.synchronize(device)
    times_sht.append(start_time.elapsed_time(end_time))


# Timing SHTCUDA on batch data
times_sht_batch = []
for _ in range(15):
    start_time.record()
    _ = sht(data_batch)
    end_time.record()
    torch.cuda.synchronize(device)
    times_sht_batch.append(start_time.elapsed_time(end_time))

# Calculate average of the last 10 timings
avg_time_sht = np.mean(times_sht[5:])
avg_time_sht_batch = np.mean(times_sht_batch[5:])/(nbatch)

print(f"Average time for non-batch SHTCUDA: {avg_time_sht:.6f} ms")
print(f"Average time for batch SHTCUDA: {avg_time_sht_batch:.6f} ms")
