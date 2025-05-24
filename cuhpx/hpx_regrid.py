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

import torch_harmonics

import cuhpx


class Grid:

    def __init__(self, grid, grid_size, norm="ortho", csphase=True):

        self.grid = grid
        self.nside = None
        self.nlat = None
        self.nlon = None
        self.norm = norm
        self.csphase = csphase

        if self.grid == 'healpix':
            self.nside = grid_size
        elif self.grid in ['legendre-gauss', 'lobatto', 'equiangular']:
            self.nlat, self.nlon = grid_size
        else:
            raise (ValueError("Unknown quadrature mode"))


class Regridding:
    def __init__(self, src_grid, dest_grid, lmax=None, mmax=None, device=None, enable_cuda=True):
        self.lmax = lmax or self._determine_lmax(src_grid, dest_grid)
        self.mmax = mmax or self._determine_mmax(src_grid, dest_grid)
        self.device = device
        self.enable_cuda = enable_cuda  # if True, use cuhpx.SHTCUDA than cuhpx.SHT

        self.sht = self._initialize_sht(src_grid, self.lmax, self.mmax).to(device)
        self.isht = self._initialize_isht(dest_grid, self.lmax, self.mmax).to(device)

    def _determine_lmax(self, src_grid, dest_grid):
        src_lmax = 2 * src_grid.nside + 1 if src_grid.nside else src_grid.nlat
        dest_lmax = 2 * dest_grid.nside + 1 if dest_grid.nside else dest_grid.nlat
        return min(src_lmax, dest_lmax)

    def _determine_mmax(self, src_grid, dest_grid):
        src_mmax = 2 * src_grid.nside + 1 if src_grid.nside else src_grid.nlon // 2 + 1
        dest_mmax = 2 * dest_grid.nside + 1 if dest_grid.nside else dest_grid.nlon // 2 + 1
        return min(src_mmax, dest_mmax)

    def _initialize_sht(self, grid, lmax, mmax):
        if grid.grid == 'healpix':

            if self.enable_cuda:
                return cuhpx.SHTCUDA(grid.nside, lmax=lmax, mmax=mmax, norm=grid.norm, csphase=grid.csphase)
            else:
                return cuhpx.SHT(grid.nside, lmax=lmax, mmax=mmax, norm=grid.norm, csphase=grid.csphase)

        elif grid.grid in ['legendre-gauss', 'lobatto', 'equiangular']:
            return torch_harmonics.RealSHT(
                grid.nlat, grid.nlon, lmax=lmax, mmax=mmax, norm=grid.norm, csphase=grid.csphase
            )
        else:
            raise ValueError("Unknown quadrature mode")

    def _initialize_isht(self, grid, lmax, mmax):
        if grid.grid == 'healpix':

            if self.enable_cuda:
                return cuhpx.iSHTCUDA(grid.nside, lmax=lmax, mmax=mmax, norm=grid.norm, csphase=grid.csphase)
            else:
                return cuhpx.iSHT(grid.nside, lmax=lmax, mmax=mmax, norm=grid.norm, csphase=grid.csphase)

        elif grid.grid in ['legendre-gauss', 'lobatto', 'equiangular']:
            return torch_harmonics.InverseRealSHT(
                grid.nlat, grid.nlon, lmax=lmax, mmax=mmax, norm=grid.norm, csphase=grid.csphase
            )
        else:
            raise ValueError("Unknown quadrature mode")

    def execute(self, f):
        return self.isht(self.sht(f))
