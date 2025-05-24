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


def random_fill_matrix(n, xmax, matrix):
    random.seed(42)
    for _ in range(n):
        v = random.random()  # Generate a random float number between 0 and 1
        x = random.randint(0, xmax - 1)  # Generate a random int number, 0 <= x < xmax
        y = random.randint(0, x)  # Generate a random int number, 0 <= y <= x
        matrix[x, y] = v  # Fill the matrix at position (x, y) with the value v

    return matrix


def generate_xyv(n, xmax, xmin):
    random.seed(42)
    v, x, y = [], [], []
    for _ in range(n):
        vi = random.random()  # Generate a random float number between 0 and 1
        xi = random.randint(xmin, xmax - 1)  # Generate a random int number, 0 <= x < xmax
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


xmax = 64
xmin = 0
xg, yg, vg = generate_xyv(100, xmax, xmin)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the range of nside values
nside_values = [32, 64, 128, 256, 512, 1024]

for nside in nside_values:
    npix = 12 * nside**2
    quad_weights = "ring"

    lmax = xmax + 1
    mmax = lmax

    sht = SHT(nside, lmax=lmax, mmax=mmax, quad_weights=quad_weights).to(device)
    isht = iSHT(nside, lmax=lmax, mmax=mmax).to(device)

    coeff_ori = torch.zeros((lmax, mmax), dtype=torch.complex128)
    coeff_ori = fill_matrix(xg, yg, vg, coeff_ori).to(device)

    signal_ori = isht(coeff_ori)

    diff = isht(sht(signal_ori)) - signal_ori
    rms = torch.sqrt((diff.abs().pow(2)).mean())
    max_value = torch.max(diff.abs())

    print(f'nside={nside}, round trip error rms = {rms}, max difference = {max_value}')
