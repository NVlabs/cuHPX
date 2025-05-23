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
import cuhpx
import numpy as np
import healpy as hp
from cuhpx import SHT, iSHT

def random_fill_matrix(n, xmax, matrix):
    random.seed(42) 
    for _ in range(n):
        v = random.random()  # Generate a random float number between 0 and 1
        x = random.randint(0, xmax-1)  # Generate a random int number, 0 <= x < xmax
        y = random.randint(0, x)  # Generate a random int number, 0 <= y <= x
        matrix[x, y] = v  # Fill the matrix at position (x, y) with the value v

    return matrix

def generate_xyv(n, xmax, xmin):
    random.seed(42) 
    v, x, y = [], [], []
    for _ in range(n):
        vi = random.random() - 0.5  # Generate a random float number between 0 and 1
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


def flm_shape(L: int) -> tuple[int, int]:

    return L, 2 * L - 1

def hp_getidx(L: int, el: int, m: int) -> int:

    return m * (2 * L - 1 - m) // 2 + el

def flm_hp_to_2d(flm_hp: np.ndarray, L: int) -> np.ndarray:

    flm_2d = np.zeros(flm_shape(L), dtype=np.complex128)

    if len(flm_hp.shape) != 1:
        raise ValueError(f"Healpix indexed flms are not flat")

    for el in range(L):
        flm_2d[el, L - 1 + 0] = flm_hp[hp_getidx(L, el, 0)]
        for m in range(1, el + 1):
            flm_2d[el, L - 1 + m] = flm_hp[hp_getidx(L, el, m)]
            flm_2d[el, L - 1 - m] = (-1) ** m * np.conj(flm_2d[el, L - 1 + m])

    return flm_2d


xmax = 60
xmin = 0
xg, yg, vg = generate_xyv(60, xmax, xmin)

nside_values = [32, 64, 128, 256, 512]
lmax = 64
mmax = lmax
quad_weights = "ring"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for nside in nside_values:
    npix = 12 * nside**2

    sht = SHT(nside, lmax=lmax, mmax=mmax, quad_weights=quad_weights).to(device)
    isht = iSHT(nside, lmax=lmax, mmax=mmax).to(device)

    coeff_ori = torch.zeros((lmax, mmax), dtype=torch.complex128)
    coeff_ori[0][0] = 1 + 0j
    coeff_ori = fill_matrix(xg, yg, vg, coeff_ori).to(device)

    signal_ori = isht(coeff_ori)

    # Perform the operation using healpy
    c = hp.map2alm(signal_ori.to('cpu').numpy(), lmax=lmax, use_weights=True)
    d = flm_hp_to_2d(c, lmax + 1)
    d = d[:-1, lmax:-1]

    # Compute the difference
    diff = torch.from_numpy(d).abs() - sht(signal_ori).to('cpu').abs()

    rms = torch.sqrt((diff.pow(2)).mean())
    max_value = torch.max(diff.abs())

    print(f'nside={nside}, difference between healpy SHT and cuhpx: rms = {rms}, max difference = {max_value}')