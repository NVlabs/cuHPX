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

import os

import numpy as np
import pkg_resources
import torch
from astropy.io import fits

DATAPATH = None


def get_datapath():
    global DATAPATH
    if DATAPATH is None:
        # Using pkg_resources to ensure the path is correctly resolved for installed packages
        DATAPATH = pkg_resources.resource_filename('cuhpx', 'data')
    return DATAPATH


def read_ring_weight(nside):
    # Use get_datapath to obtain the correct data directory
    data_path = get_datapath()
    filename = f"weight_ring_n{nside:05d}.fits"
    weightfile = os.path.join(data_path, filename)

    with fits.open(weightfile) as inp:
        inp_hdu = inp[1]
        weight = inp_hdu.data.field(0)
        weight = weight.flatten()
    return weight


def apply_ring_weight(nside):
    w = read_ring_weight(nside)
    pi = np.pi
    npix = nside * nside * 12
    nrings = 4 * nside - 1  # Total rings
    weights = np.zeros(nrings)

    for m in range(nrings):
        ring = m + 1
        northring = 4 * nside - ring if ring > 2 * nside else ring
        weights[m] = 4.0 * pi / npix * (1.0 + w[northring - 1])

    return weights


def load_pixel_weights(dir, nside):

    file_name = f'{dir}/weight_pixel_n{nside:05d}.fits'

    with fits.open(file_name) as hdul:
        table_data = hdul[1].data
        weights = table_data['COMPRESSED PIXEL WEIGHTS']

    weights = weights.byteswap().newbyteorder()
    return weights


def apply_pixel_weight(pix, w, setwgt):

    if setwgt:
        pix = w
    else:
        pix *= 1.0 + w
    return pix


def apply_full_weights(map_hp, setwgt, nside):

    npix = 12 * nside**2
    map_hp = np.ones(npix)
    wgt = load_pixel_weights('data', nside)
    pix = 0
    vpix = 0

    for i in range(2 * nside):
        shifted = (i < nside - 1) or ((i + nside) & 1)
        qpix = min(nside, i + 1)
        odd = qpix & 1
        wpix = ((qpix + 1) >> 1) + (0 if (odd or shifted) else 1)
        psouth = npix - pix - (qpix << 2)

        for j in range(qpix << 2):
            j4 = j % qpix
            rpix = min(j4, qpix - (1 if shifted else 0) - j4)
            pix_value = apply_pixel_weight(map_hp[pix + j], wgt[vpix + rpix], setwgt)
            map_hp[pix + j] = pix_value

            if i != 2 * nside - 1:
                pix_value = apply_pixel_weight(map_hp[psouth + j], wgt[vpix + rpix], setwgt)
                map_hp[psouth + j] = pix_value

        pix += qpix << 2
        vpix += wpix

    return map_hp


def healpix_weights(nside: int, weight) -> np.ndarray:

    # Convert (ring) index to :math:`\theta` angle for HEALPix sampling scheme.

    t = np.arange(4 * nside - 1)
    z = np.zeros_like(t, dtype=float)

    npix = 12 * nside**2

    if weight == 'ring':
        w = apply_ring_weight(nside)
    else:
        w = 4.0 * np.pi / npix * np.ones_like(t)

    # Define masks
    mask1 = t < (nside - 1)
    mask2 = (t >= (nside - 1)) & (t <= (3 * nside - 1))
    mask3 = (t > (3 * nside - 1)) & (t <= (4 * nside - 2))

    # Update z with conditions
    z[mask1] = 1 - ((t[mask1] + 1) ** 2) / (3 * nside**2)
    z[mask2] = 4 / 3 - 2 * (t[mask2] + 1) / (3 * nside)
    z[mask3] = ((4 * nside - 1 - t[mask3]) ** 2) / (3 * nside**2) - 1

    z = np.flip(z)

    return z, w


def ftm_shape(L: int, sampling: str, nside: int) -> tuple[int, int]:

    # Shape of intermediate array
    # L: harmonic band-limit
    if sampling.lower() != "healpix":
        raise ValueError(f"sampling can only be 'healpix', '{sampling}' unsupported")
        # return 4*nside-1, 2*L
    return 4 * nside - 1, L


def f_shape(sampling: str, nside: int) -> tuple[int]:

    # shape of the spherical signal
    if sampling.lower() != "healpix":
        raise ValueError(f"sampling can only be 'healpix', '{sampling}' unsupported")
    return (12 * nside**2,)


def p2phi_ring(t: int, p: int, nside: int) -> np.ndarray:

    # Convert index to phi angle for HEALPix
    # t: theta, index of ring
    # p: phi, index within ring

    shift = 1 / 2
    if (t + 1 >= nside) & (t + 1 <= 3 * nside):
        shift *= (t - nside + 2) % 2
        factor = np.pi / (2 * nside)
        return factor * (p + shift)
    elif t + 1 > 3 * nside:
        factor = np.pi / (2 * (4 * nside - t - 1))
    else:
        factor = np.pi / (2 * (t + 1))
    return factor * (p + shift)


def nphi_ring(t: int, nside: int) -> int:

    # Number of phi samples for HEALPix sampling on give theta ring
    if (t >= 0) and (t < nside - 1):
        return 4 * (t + 1)
    elif (t >= nside - 1) and (t <= 3 * nside - 1):
        return 4 * nside
    elif (t > 3 * nside - 1) and (t <= 4 * nside - 2):
        return 4 * (4 * nside - t - 1)
    else:
        raise ValueError(f"Ring t={t} not contained by nside={nside}")


def legpoly(mmax, lmax, x, norm="ortho", inverse=False, csphase=True):

    nmax = max(mmax, lmax)
    vdm = np.zeros((nmax, nmax, len(x)), dtype=np.float64)

    norm_factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor

    # initial values to start the recursion
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    # fill the diagonal and the lower diagonal
    for l in range(1, nmax):  # noqa: E741
        vdm[l - 1, l, :] = np.sqrt(2 * l + 1) * x * vdm[l - 1, l - 1, :]
        vdm[l, l, :] = np.sqrt((2 * l + 1) * (1 + x) * (1 - x) / 2 / l) * vdm[l - 1, l - 1, :]

    # fill the remaining values on the upper triangle and multiply b
    for l in range(2, nmax):  # noqa: E741
        for m in range(0, l - 1):
            vdm[m, l, :] = (
                x * np.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m)) * vdm[m, l - 1, :]
                - np.sqrt((l + m - 1) / (l - m) * (2 * l + 1) / (2 * l - 3) * (l - m - 1) / (l + m)) * vdm[m, l - 2, :]
            )

    if norm == "schmidt":
        for l in range(0, nmax):  # noqa: E741
            if inverse:
                vdm[:, l, :] = vdm[:, l, :] * np.sqrt(2 * l + 1)
            else:
                vdm[:, l, :] = vdm[:, l, :] / np.sqrt(2 * l + 1)

    vdm = vdm[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            vdm[m] *= 1

    return vdm


def legpoly_torch(mmax, lmax, x):

    nmax = max(mmax, lmax)
    vdm = torch.zeros((nmax, nmax, len(x)), dtype=torch.float64)

    norm_factor = 1.0

    # Initial values to start the recursion
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    # Fill the diagonal and the lower diagonal
    for l in range(1, nmax):  # noqa: E741
        vdm[l - 1, l, :] = np.sqrt(2 * l + 1) * x * vdm[l - 1, l - 1, :]
        vdm[l, l, :] = np.sqrt((2 * l + 1) * (1 + x) * (1 - x) / 2 / l) * vdm[l - 1, l - 1, :]

    # Fill the remaining values on the upper triangle
    for l in range(2, nmax):  # noqa: E741
        for m in range(0, l - 1):
            factor1 = np.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m))
            factor2 = np.sqrt((l + m - 1) / (l - m) * (2 * l + 1) / (2 * l - 3) * (l - m - 1) / (l + m))
            vdm[m, l, :] = x * factor1 * vdm[m, l - 1, :] - factor2 * vdm[m, l - 2, :]

    vdm = vdm[:mmax, :lmax, :]

    return vdm


def _precompute_legpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    return legpoly(mmax, lmax, np.cos(t), norm=norm, inverse=inverse, csphase=csphase)


def _precompute_legpoly_torch(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    return legpoly_torch(mmax, lmax, torch.from_numpy(np.cos(t)))


def _precompute_dlegpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):

    pct = _precompute_legpoly(mmax + 1, lmax + 1, t, norm=norm, inverse=inverse, csphase=False)

    dpct = np.zeros((2, mmax, lmax, len(t)), dtype=np.float64)

    # fill the derivative terms wrt theta
    for l in range(0, lmax):  # noqa: E741

        # m = 0
        dpct[0, 0, l] = -np.sqrt(l * (l + 1)) * pct[1, l]

        # 0 < m < l
        for m in range(1, min(l, mmax)):
            dpct[0, m, l] = 0.5 * (
                np.sqrt((l + m) * (l - m + 1)) * pct[m - 1, l] - np.sqrt((l - m) * (l + m + 1)) * pct[m + 1, l]
            )

        # m == l
        if mmax > l:
            dpct[0, l, l] = np.sqrt(l / 2) * pct[l - 1, l]

        # fill the - 1j m P^m_l / sin(phi). as this component is purely imaginary,
        for m in range(1, min(l + 1, mmax)):
            # this component is implicitly complex
            dpct[1, m, l] = (
                0.5
                * np.sqrt((2 * l + 1) / (2 * l + 3))
                * (
                    np.sqrt((l - m + 1) * (l - m + 2)) * pct[m - 1, l + 1]
                    + np.sqrt((l + m + 1) * (l + m + 2)) * pct[m + 1, l + 1]
                )
            )
    if csphase:
        for m in range(1, mmax, 2):
            dpct[:, m] *= -1

    return dpct


def W_helper(w, nside):

    W = torch.zeros(f_shape(sampling="healpix", nside=nside), dtype=torch.float)
    ntheta = 4 * nside - 1
    index = 0
    for t in range(ntheta):
        nphi = nphi_ring(t, nside)
        W[index : index + nphi] = w[t] / 2
        index += nphi
    return W


def cumulative_nphi_ring(t, nside):
    if 0 <= t < nside:
        return 2 * t * (t + 1)
    elif t < 3 * nside:
        northern_sum = 2 * nside * (nside + 1)
        equatorial_count = (t - nside) * 4 * nside
        return northern_sum + equatorial_count
    elif t < 4 * nside:
        total_sum = 12 * nside * nside
        remaining_rings = 4 * nside - t - 1
        remaining_sum = 2 * remaining_rings * (remaining_rings + 1)
        return total_sum - remaining_sum
    else:
        return -1  # Error case
