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

import healpy as hp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from download import download_file

from cuhpx import iSHTCUDA as iSHT

download_file('http://lambda.gsfc.nasa.gov/data/map/dr4/skymaps/7yr/raw/wmap_band_iqumap_r9_7yr_W_v4.fits')

nside = int(input("Enter the nside value: "))
lmax = int(input("Enter the lmax value: "))  # lmax = 2*nside+1
mmax = lmax
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wmap_map_I = hp.read_map("wmap_band_iqumap_r9_7yr_W_v4.fits")
wmap = hp.ud_grade(wmap_map_I, nside)
data = torch.from_numpy(wmap)
signal = data.to(device)


class SpectralModel(nn.Module):
    def __init__(self, nside, lmax, mmax):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(lmax, mmax, dtype=torch.complex128))
        self.isht = iSHT(nside, lmax=lmax, mmax=mmax).to(device)

    def forward(self):
        return self.isht(self.coeffs)


sh_model = SpectralModel(nside, lmax, mmax).to(device)

optimizer = torch.optim.Adam(sh_model.parameters(), lr=5e-2)

losses = []


def plot_and_save_map(output_map, iter_num):
    plt.figure(figsize=(10, 8))
    hp.mollview(
        output_map.detach().cpu().numpy(),  # Detach before converting to numpy
        coord=["G", "E"],
        title=f"Iteration {iter_num}: Histogram equalized Ecliptic",
        unit="mK",
        norm="hist",
        min=-1,
        max=1,
    )
    hp.graticule()
    plt.savefig(f'output_map_iter_{iter_num}.png')
    plt.close()


for iter in range(500):
    output_map = sh_model()
    loss = (output_map - signal).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iter % 10 == 0:
        print(f'Iteration: {iter} Loss: {loss.item()}')
        plot_and_save_map(output_map, iter)
