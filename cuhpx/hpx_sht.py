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

import cuhpx_fft
import numpy as np
import torch
import torch.cuda.nvtx
import torch.nn as nn
from torch.autograd import Function

from cuhpx.sht_tools import (
    W_helper,
    _precompute_dlegpoly,
    _precompute_legpoly,
    _precompute_legpoly_torch,
    cumulative_nphi_ring,
    f_shape,
    ftm_shape,
    healpix_weights,
    nphi_ring,
    p2phi_ring,
)


def healpix_rfft_torch(f: torch.tensor, L: int, nside: int) -> torch.tensor:

    index = 0
    ctype = torch.complex64 if f.dtype == torch.float32 else torch.complex128
    ftm = torch.zeros(ftm_shape(L, "healpix", nside), dtype=ctype, device=f.device)
    ntheta = ftm.shape[0]
    for t in range(ntheta):
        nphi = nphi_ring(t, nside)

        fm_chunk = torch.fft.rfft(f[index : index + nphi], norm="backward")  # backward
        ftm[t, : min(nphi // 2 + 1, L)] = fm_chunk[: min(nphi // 2 + 1, L)]

        index += nphi

        phi_ring_offset = p2phi_ring(t, 0, nside)
        phase_shift = torch.exp(-1j * torch.arange(L, device=f.device) * phi_ring_offset)
        ftm[t, :] *= phase_shift

    return ftm


def healpix_irfft_torch(ftm: torch.tensor, L: int, nside: int) -> torch.tensor:

    ftype = torch.float if ftm.dtype == torch.complex64 else torch.double
    f = torch.zeros(f_shape(sampling="healpix", nside=nside), dtype=ftype, device=ftm.device)
    ntheta = ftm.shape[0]
    index = 0
    for t in range(ntheta):

        phi_ring_offset = p2phi_ring(t, 0, nside)
        phase_shift = torch.exp(1j * torch.arange(L, device=ftm.device) * phi_ring_offset)
        ftm[t, :] *= phase_shift

        nphi = nphi_ring(t, nside)

        fm_chunk = ftm[t, :]
        f[index : index + nphi] = torch.fft.irfft(fm_chunk, n=nphi, norm="forward")

        index += nphi
    return f


def healpix_irfft_bluestein(ftm: torch.tensor, L: int, nside: int) -> torch.tensor:

    f = torch.zeros(12 * nside**2, dtype=torch.double, device=ftm.device)

    ntheta = ftm.shape[0]
    padding = 8 * nside

    x_pad = torch.zeros(ntheta, padding, dtype=torch.complex128, device=ftm.device)
    y_pad = torch.zeros(ntheta, padding, dtype=torch.complex128, device=ftm.device)

    for t in range(ntheta):
        phi_ring_offset = p2phi_ring(t, 0, nside)
        phase_shift = torch.exp(1j * torch.arange(L, device=ftm.device) * phi_ring_offset)
        ftm[t, :] *= phase_shift

    for t in range(ntheta):
        nphi = nphi_ring(t, nside)
        index = cumulative_nphi_ring(t, nside)

        fm_chunk = torch.zeros(nphi // 2 + 1, dtype=torch.complex128).to(ftm.device)
        fm_chunk[: min(nphi // 2 + 1, L)] = ftm[t, : min(nphi // 2 + 1, L)]
        fm_chunk = torch.cat((fm_chunk, fm_chunk[1:-1].conj().flip(0))).conj()

        coef_arr = 1j * torch.pi * (torch.arange(nphi) ** 2) / nphi

        chirp_a = torch.exp(coef_arr).to(ftm.device)
        chirp_b = torch.exp(-coef_arr).to(ftm.device)

        x_pad[t, :nphi] = fm_chunk * chirp_b
        y_pad[t, :nphi] = chirp_a
        y_pad[t, padding - nphi + 1 :] = torch.flip(chirp_a[1:], dims=[0])

    # Conv
    x_pad = torch.fft.fft(x_pad, dim=-1)
    y_pad = torch.fft.fft(y_pad, dim=-1)
    x_pad *= y_pad
    x_pad = torch.fft.ifft(x_pad, dim=-1)

    for t in range(ntheta):
        nphi = nphi_ring(t, nside)
        index = cumulative_nphi_ring(t, nside)
        coef_arr = 1j * torch.pi * (torch.arange(nphi) ** 2) / nphi
        chirp_b = torch.exp(-coef_arr).to(ftm.device)
        result = x_pad[t, :nphi] * chirp_b
        f[index : index + nphi] = result.real

    return f


def healpix_rfft_bluestein(f: torch.tensor, L: int, nside: int) -> torch.tensor:

    ftm = torch.zeros((4 * nside - 1, L), dtype=torch.complex128, device=f.device)
    ntheta = ftm.shape[0]

    padding = 8 * nside

    x_pad = torch.zeros(ntheta, padding, dtype=torch.complex128, device=f.device)
    y_pad = torch.zeros(ntheta, padding, dtype=torch.complex128, device=f.device)

    for t in range(ntheta):
        nphi = nphi_ring(t, nside)
        index = cumulative_nphi_ring(t, nside)

        vec = f[index : index + nphi]
        coef_arr = 1j * torch.pi * (torch.arange(nphi) ** 2) / nphi

        chirp_b = torch.exp(coef_arr).to(f.device)
        chirp_a = 1 / chirp_b

        x_pad[t, :nphi] = vec * chirp_b
        y_pad[t, :nphi] = chirp_a
        y_pad[t, padding - nphi + 1 :] = torch.flip(chirp_a[1:], dims=[0])

    # Conv
    x_pad = torch.fft.fft(x_pad, dim=-1)
    y_pad = torch.fft.fft(y_pad, dim=-1)
    x_pad *= y_pad
    x_pad = torch.fft.ifft(x_pad, dim=-1)

    for t in range(ntheta):
        nphi = nphi_ring(t, nside)
        coef_arr = 1j * torch.pi * (torch.arange(nphi) ** 2) / nphi
        chirp_b = torch.exp(coef_arr).to(f.device)
        result = (x_pad[t, :nphi] * chirp_b).conj()
        ftm[t, : min(L, nphi // 2 + 1)] = result[: min(L, nphi // 2 + 1)]

    for t in range(ntheta):
        phi_ring_offset = p2phi_ring(t, 0, nside)
        phase_shift = torch.exp(-1j * torch.arange(L, device=f.device) * phi_ring_offset)
        ftm[t, :] *= phase_shift

    return ftm


class SHT(nn.Module):

    def __init__(
        self,
        nside,
        lmax=None,
        mmax=None,
        grid="healpix",
        quad_weights="ring",
        norm="ortho",
        csphase=True,
        use_bluestein=False,
    ):

        super().__init__()

        self.nside = nside
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside
        self.quad_weights = quad_weights
        self.use_bluestein = use_bluestein

        if self.grid == "healpix":
            cost, w = healpix_weights(nside, self.quad_weights)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        tq = np.flip(np.arccos(cost))
        self.mmax = mmax or (self.nlon // 2 + 1)
        weights = torch.from_numpy(w)

        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = torch.from_numpy(pct)

        weights = torch.einsum('mlk,k->mlk', pct, weights)
        self.register_buffer('weights', weights, persistent=False)

    def forward(self, x: torch.Tensor):

        if torch.is_complex(x):
            raise ValueError("Input tensor must be real.")

        if self.use_bluestein:
            x = healpix_rfft_bluestein(x, self.mmax, self.nside)
        else:
            x = healpix_rfft_torch(x, self.mmax, self.nside)
        x = torch.view_as_real(x)

        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax

        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction
        xout[..., 0] = torch.einsum('...km,mlk->...lm', x[..., : self.mmax, 0], self.weights.to(x.dtype))
        xout[..., 1] = torch.einsum('...km,mlk->...lm', x[..., : self.mmax, 1], self.weights.to(x.dtype))
        x = torch.view_as_complex(xout)

        return x


class iSHT(nn.Module):

    def __init__(self, nside, lmax=None, mmax=None, grid="healpix", norm="ortho", csphase=True, use_bluestein=False):

        super().__init__()

        self.nside = nside
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside

        self.use_bluestein = use_bluestein

        if self.grid == "healpix":
            cost, _ = healpix_weights(nside, 'none')
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        t = np.flip(np.arccos(cost))

        self.mmax = mmax or (self.nlon // 2 + 1)
        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        pct = torch.from_numpy(pct)

        self.register_buffer('pct', pct, persistent=False)

    def forward(self, x: torch.Tensor):

        x = torch.view_as_real(x)

        rl = torch.einsum('...lm, mlk->...km', x[..., 0], self.pct.to(x.dtype))
        im = torch.einsum('...lm, mlk->...km', x[..., 1], self.pct.to(x.dtype))
        xs = torch.stack((rl, im), -1)

        x = torch.view_as_complex(xs)

        if self.use_bluestein:
            x = healpix_irfft_bluestein(x, self.mmax, self.nside)
        else:
            x = healpix_irfft_torch(x, self.mmax, self.nside)

        return x


class VectorSHT(nn.Module):

    def __init__(
        self,
        nside,
        lmax=None,
        mmax=None,
        grid="healpix",
        quad_weights="ring",
        norm="ortho",
        csphase=True,
        use_bluestein=False,
    ):

        super().__init__()

        self.nside = nside
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside
        self.quad_weights = quad_weights
        self.use_bluestein = use_bluestein

        if self.grid == "healpix":
            cost, w = healpix_weights(nside, self.quad_weights)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        tq = np.flip(np.arccos(cost))
        self.mmax = mmax or (self.nlon // 2 + 1)
        weights = torch.from_numpy(w)

        dpct = _precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        dpct = torch.from_numpy(dpct)

        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)  # noqa: E741
        norm_factor = 1.0 / l / (l + 1)
        norm_factor[0] = 1.0
        weights = torch.einsum('dmlk,k,l->dmlk', dpct, weights, norm_factor)

        weights[1] = -1 * weights[1]

        self.register_buffer('weights', weights, persistent=False)

    def forward(self, x: torch.Tensor):

        if torch.is_complex(x):
            raise ValueError("Input tensor must be real.")

        # x = healpix_rfft_torch(x, self.mmax, self.nside)

        if x.dim() == 1:
            x = cuhpx_fft.healpix_rfft_class(x, self.mmax, self.nside)
        else:
            x = cuhpx_fft.healpix_rfft_batch(x, self.mmax, self.nside)

        x = torch.view_as_real(x)
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax

        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction - spheroidal component
        # real component
        xout[..., 0, :, :, 0] = torch.einsum(
            '...km,mlk->...lm', x[..., 0, :, : self.mmax, 0], self.weights[0].to(x.dtype)
        ) - torch.einsum('...km,mlk->...lm', x[..., 1, :, : self.mmax, 1], self.weights[1].to(x.dtype))

        # iamg component
        xout[..., 0, :, :, 1] = torch.einsum(
            '...km,mlk->...lm', x[..., 0, :, : self.mmax, 1], self.weights[0].to(x.dtype)
        ) + torch.einsum('...km,mlk->...lm', x[..., 1, :, : self.mmax, 0], self.weights[1].to(x.dtype))

        # contraction - toroidal component
        # real component
        xout[..., 1, :, :, 0] = -torch.einsum(
            '...km,mlk->...lm', x[..., 0, :, : self.mmax, 1], self.weights[1].to(x.dtype)
        ) - torch.einsum('...km,mlk->...lm', x[..., 1, :, : self.mmax, 0], self.weights[0].to(x.dtype))
        # imag component
        xout[..., 1, :, :, 1] = torch.einsum(
            '...km,mlk->...lm', x[..., 0, :, : self.mmax, 0], self.weights[1].to(x.dtype)
        ) - torch.einsum('...km,mlk->...lm', x[..., 1, :, : self.mmax, 1], self.weights[0].to(x.dtype))

        return torch.view_as_complex(xout)


class VectoriSHT(nn.Module):

    def __init__(self, nside, lmax=None, mmax=None, grid="healpix", norm="ortho", csphase=True, use_bluestein=False):

        super().__init__()

        self.nside = nside
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside

        self.use_bluestein = use_bluestein

        if self.grid == "healpix":
            cost, _ = healpix_weights(nside, 'none')
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        t = np.flip(np.arccos(cost))

        self.mmax = mmax or (self.nlon // 2 + 1)
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        dpct = torch.from_numpy(dpct)

        self.register_buffer('dpct', dpct, persistent=False)

    def forward(self, x: torch.Tensor):

        x = torch.view_as_real(x)

        # contraction - spheroidal component
        # real component
        srl = torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[0].to(x.dtype)) - torch.einsum(
            '...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[1].to(x.dtype)
        )
        # iamg component
        sim = torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[0].to(x.dtype)) + torch.einsum(
            '...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[1].to(x.dtype)
        )

        # contraction - toroidal component
        # real component
        trl = -torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[1].to(x.dtype)) - torch.einsum(
            '...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[0].to(x.dtype)
        )
        # imag component
        tim = torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[1].to(x.dtype)) - torch.einsum(
            '...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[0].to(x.dtype)
        )

        # reassemble
        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)

        x = torch.view_as_complex(xs)

        # x = healpix_irfft_torch(x, self.mmax, self.nside)

        if x.dim() == 2:
            x = cuhpx_fft.healpix_irfft_class(x, self.mmax, self.nside)
        else:
            x = cuhpx_fft.healpix_irfft_batch(x, self.mmax, self.nside)

        return x


def einsum_with_chunking(x, weights, mmax, xout, nchunk, stream1):

    device = torch.device("cuda")
    chunk_size = int(weights.size(1) / nchunk + 1)  # Adjust this based on your memory constraints

    torch.cuda.nvtx.range_push("Allocate memory for chunk")
    next_chunk_cpu = torch.empty((weights.size(0), chunk_size, weights.size(2)), dtype=weights.dtype, pin_memory=True)
    current_chunk = torch.empty((weights.size(0), chunk_size, weights.size(2)), dtype=weights.dtype, device=device)
    next_chunk = torch.empty_like(current_chunk)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("einsum between x and weights with chunking")

    # Create events for synchronization
    event_transfer = torch.cuda.Event(blocking=True)
    event_computation = torch.cuda.Event(blocking=True)
    start_j, end_j = 0, 0

    torch.cuda.current_stream().synchronize()

    for i in range(0, weights.size(1), chunk_size):

        start_i, end_i = i, min(i + chunk_size, weights.size(1))
        actual_chunk_size = end_i - start_i

        # torch.cuda.current_stream().synchronize()
        event_transfer.synchronize()

        if actual_chunk_size != chunk_size:
            next_chunk_cpu.resize_((weights.size(0), actual_chunk_size, weights.size(2)))

        torch.cuda.nvtx.range_push("CPU copy from weights to pin memory")
        next_chunk_cpu.copy_(weights[:, start_i:end_i, :])
        torch.cuda.nvtx.range_pop()

        with torch.cuda.stream(stream1):
            torch.cuda.nvtx.range_push(f"Transfer weights chunk {i}:{end_i} to GPU")
            next_chunk[: weights.size(0), : end_i - start_i, :].copy_(next_chunk_cpu, non_blocking=True)
            event_transfer.record(stream1)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"Compute einsum for chunk {i - chunk_size}:{end_i - chunk_size}")
        xout[..., start_j:end_j, :, :] = torch.einsum(
            '...kmn,mlk->...lmn', x, current_chunk[:, : end_j - start_j, :].to(x.dtype)
        )

        event_computation.record(torch.cuda.current_stream())
        torch.cuda.nvtx.range_pop()
        torch.cuda.current_stream().wait_event(event_transfer)

        current_chunk, next_chunk = next_chunk, current_chunk
        start_j, end_j = start_i, end_i

    if start_i < weights.size(1):
        torch.cuda.nvtx.range_push("Compute einsum for the last chunk")
        xout[..., start_i:end_i, :, :] = torch.einsum(
            '...kmn,mlk->...lmn', x, current_chunk[:, : end_i - start_i, :].to(x.dtype)
        )
        torch.cuda.nvtx.range_pop()

    stream1.synchronize()
    torch.cuda.current_stream().synchronize()

    torch.cuda.nvtx.range_pop()  # End of einsum with chunking

    return xout


class SHTFunction(Function):

    @staticmethod
    def forward(ctx, x, weights, pct, W, mmax, lmax, nside):

        # Init
        ctx.save_for_backward(pct, W)
        ctx.mmax = mmax
        ctx.lmax = lmax
        ctx.nside = nside

        torch.cuda.nvtx.range_push("rfft")
        # SHT
        if x.dim() == 1:
            x = cuhpx_fft.healpix_rfft_class(x, mmax, nside)
        else:
            x = cuhpx_fft.healpix_rfft_batch(x, mmax, nside)

        torch.cuda.nvtx.range_pop()

        x = torch.view_as_real(x)

        out_shape = list(x.size())
        out_shape[-3] = lmax
        out_shape[-2] = mmax

        torch.cuda.nvtx.range_push("allocate xout")
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("einsum between pct and weights")
        weights = pct * weights
        torch.cuda.nvtx.range_pop()

        if not pct.is_cuda:
            torch.cuda.nvtx.range_push("einsum between x and weights using two stream")
            nchunk = 12
            stream1 = torch.cuda.Stream()
            xout = einsum_with_chunking(x, weights, mmax, xout, nchunk, stream1)
            torch.cuda.nvtx.range_pop()
        else:
            torch.cuda.nvtx.range_push("einsum between x and weights")
            xout = torch.einsum('...kmn,mlk->...lmn', x, weights.to(x.dtype))
            torch.cuda.nvtx.range_pop()

        x = torch.view_as_complex(xout.contiguous())

        return x

    @staticmethod
    def backward(ctx, grad_output):

        # adjoint iSHT
        pct, W = ctx.saved_tensors
        mmax, nside = ctx.mmax, ctx.nside

        x = torch.view_as_real(grad_output)

        xs = torch.einsum('...lmn, mlk->...kmn', x, pct.to(x.dtype))
        grad_input = torch.view_as_complex(xs.contiguous())

        if grad_input.dim() == 2:
            grad_input = cuhpx_fft.healpix_irfft_class(grad_input, mmax, nside)
        else:
            grad_input = cuhpx_fft.healpix_irfft_batch(grad_input, mmax, nside)

        grad_input = grad_input * W.to(x.dtype)

        return grad_input, None, None, None, None, None, None


class iSHTFunction(Function):

    @staticmethod
    def forward(ctx, x, weights, pct, W, mmax, lmax, nside):

        ctx.save_for_backward(weights, pct, W)
        ctx.mmax = mmax
        ctx.lmax = lmax
        ctx.nside = nside

        x = torch.view_as_real(x)

        torch.cuda.nvtx.range_push("einsum between x and pct")
        xs = torch.einsum('...lmn, mlk->...kmn', x, pct.to(x.dtype))
        torch.cuda.nvtx.range_pop()

        x = torch.view_as_complex(xs.contiguous())

        torch.cuda.nvtx.range_push("irfft")
        if x.dim() == 2:
            x = cuhpx_fft.healpix_irfft_class(x, mmax, nside)
        else:
            x = cuhpx_fft.healpix_irfft_batch(x, mmax, nside)
        torch.cuda.nvtx.range_pop()

        return x

    @staticmethod
    def backward(ctx, grad_output):

        # adjoint SHT
        weights, pct, W = ctx.saved_tensors
        mmax, lmax, nside = ctx.mmax, ctx.lmax, ctx.nside

        x = grad_output / W.to(grad_output.dtype)

        if x.dim() == 1:
            x = cuhpx_fft.healpix_rfft_class(x, mmax, nside)
        else:
            x = cuhpx_fft.healpix_rfft_batch(x, mmax, nside)

        x = torch.view_as_real(x)

        out_shape = list(x.size())
        out_shape[-3] = lmax
        out_shape[-2] = mmax

        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        weights = pct * weights
        weights = weights.to(x.device)

        xout = torch.einsum('...kmn,mlk->...lmn', x, weights.to(x.dtype))
        grad_input = torch.view_as_complex(xout.contiguous())

        return grad_input, None, None, None, None, None, None


class SHTCUDA(nn.Module):
    def __init__(self, nside, lmax=None, mmax=None, quad_weights="ring", norm="ortho", csphase=True):

        super().__init__()
        self.nside = nside
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside
        self.quad_weights = quad_weights
        self.lmax = lmax or self.nlat
        self.mmax = mmax or (self.nlon // 2 + 1)
        self.stream = torch.cuda.current_stream()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is not available. This class requires a GPU.")

        self.device = torch.device('cuda')

        # quadrature weights
        cost, w = healpix_weights(nside, self.quad_weights)
        tq = np.flip(np.arccos(cost))
        weights = torch.from_numpy(w)
        weights = weights.to(torch.float)

        # Legendre polynomials
        pct = _precompute_legpoly_torch(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = pct.to(torch.float)

        if not (min(self.nside, self.lmax, self.mmax) > 2**9):
            weights = weights.to(self.device)
            pct = pct.to(self.device)

        # W for adjoint SHT for graident evaluation
        W = W_helper(w, nside)
        W = W.to(torch.float).to(self.device)

        self.register_buffer('weights', weights, persistent=False)
        self.register_buffer('pct', pct, persistent=False)
        self.register_buffer('W', W, persistent=False)

    def forward(self, x):

        if torch.is_complex(x):
            raise ValueError("Input tensor must be real.")

        with torch.cuda.stream(self.stream):
            return SHTFunction.apply(x, self.weights, self.pct, self.W, self.mmax, self.lmax, self.nside)


class iSHTCUDA(nn.Module):

    def __init__(self, nside, lmax=None, mmax=None, quad_weights="ring", norm="ortho", csphase=True):

        super().__init__()
        self.nside = nside
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside
        self.quad_weights = quad_weights
        self.lmax = lmax or self.nlat
        self.mmax = mmax or (self.nlon // 2 + 1)
        self.stream = torch.cuda.current_stream()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is not available. This class requires a GPU.")

        self.device = torch.device('cuda')

        # quadrature weights
        cost, w = healpix_weights(nside, self.quad_weights)
        tq = np.flip(np.arccos(cost))
        weights = torch.from_numpy(w)
        weights = weights.to(torch.float).to(self.device)

        # Legendre polynomials
        pct = _precompute_legpoly_torch(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = pct.to(torch.float).to(self.device)

        # W for adjoint SHT for graident evaluation
        W = W_helper(w, nside)
        W = W.to(torch.float).to(self.device)

        self.register_buffer('weights', weights, persistent=False)
        self.register_buffer('pct', pct, persistent=False)
        self.register_buffer('W', W, persistent=False)

    def forward(self, x):

        with torch.cuda.stream(self.stream):
            return iSHTFunction.apply(x, self.weights, self.pct, self.W, self.mmax, self.lmax, self.nside)
