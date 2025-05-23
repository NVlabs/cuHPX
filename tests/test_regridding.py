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
from cuhpx import Grid, Regridding
import random

def random_fill_matrix(n, xmax, matrix):
    for _ in range(n):
        v = random.random()  # Generate a random float number between 0 and 1
        x = random.randint(0, xmax-1)  # Generate a random int number, 0 <= x < xmax
        y = random.randint(0, x)  # Generate a random int number, 0 <= y <= x
        matrix[x, y] = v  # Fill the matrix at position (x, y) with the value v

    return matrix

def generate_xyv(n, xmax, xmin):
    v, x, y = [], [], []
    for _ in range(n):
        vi = random.random()  # Generate a random float number between 0 and 1
        xi = random.randint(xmin, xmax-1)  # Generate a random int number, 0 <= x < xmax
        yi = random.randint(xmin, xi)  # Generate a random int number, 0 <= y <= x

        v.append(vi)
        x.append(xi)
        y.append(yi)

    return x, y, v

def fill_matrix(x, y, v, matrix):

    n = len(x)
    for i in range(n):
        matrix[x[i], y[i]] = v[i]  # Fill the matrix at position (x, y) with the value v
    return matrix

nside = int(input("Enter the nside value: "))

xmax = 2*nside-1
xmin = 0
xg, yg, vg = generate_xyv(100, xmax, xmin)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

npix = 12*nside**2

lmax = 2*nside+1
mmax = lmax

#synthetic data with finite bandwidth
sht = cuhpx.SHT(nside, lmax = lmax, mmax=mmax)
isht = cuhpx.iSHT(nside, lmax = lmax, mmax=mmax)

coeff = torch.zeros((lmax, mmax), dtype=torch.complex128)
coeff = fill_matrix(xg, yg, vg, coeff)
signal_hpx = isht(coeff).to(device)

#regridding
src_grid = Grid('healpix', nside)
dest_grid = Grid('equiangular', (2*nside,4*nside))

hpx2eq = Regridding(src_grid, dest_grid, lmax=lmax, mmax=mmax, device=device)
eq2hpx = Regridding(dest_grid, src_grid, lmax=lmax, mmax=mmax, device=device)

signal_eq = hpx2eq.execute(signal_hpx)
signal_hpx_back = eq2hpx.execute(signal_eq)

diff = signal_hpx_back - signal_hpx
rms = torch.sqrt((diff.pow(2)).mean())
max_value = torch.max(diff.abs())

print(f'regridding error: nside={nside}, rms = {rms}, max difference = {max_value}')



