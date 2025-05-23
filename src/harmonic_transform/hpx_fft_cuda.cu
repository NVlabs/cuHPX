/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <complex>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>

#ifndef MY_PI
#define MY_PI
#define PI (4 * atan(1.0))
#endif // MY_PI

template <typename T> __device__ inline c10::complex<T> conj(const c10::complex<T>& z) {
        return c10::complex<T>(z.real(), -z.imag());
    }

__device__ inline int d_nphi_ring(int t, int nside);
__device__ inline int d_cumulative_nphi_ring(int t, int nside);
template <typename T> __device__ inline T d_p2phi_ring(int t, int p, int nside);
template <typename I> __device__ inline int d_isqrt(I arg);
template <typename I> __device__ inline void ring_idx2theta_phi(const I pix, I& iring, I& iphi, const int order, const I nside);
template <typename I> __device__ inline void computer_iring_iphi_nphi(const I pix, I& iring, I& iphi, I& nphi,
        const int order, const I nside);
template <typename I> __device__ inline int compute_order(I nside);


template<typename T> __device__ inline c10::complex<T> compute_complex_exp(T coef);

template<> __device__ inline c10::complex<double> compute_complex_exp<double>(double coef) {
    return c10::complex<double>(cos(coef), sin(coef));
}

template<> __device__ inline c10::complex<float> compute_complex_exp<float>(float coef) {
    // Use expf for float;
    return c10::complex<float>(cosf(coef), sinf(coef));
}

template <typename T> __device__ inline T mod2(T value) {
    return fmod(value, 2.0);
}


template <typename T> __global__ void rfft_pre_process(c10::complex<T>* x_pad, c10::complex<T>* y_pad, const T* f, int padding, int nside){

    int ipix = blockIdx.x * blockDim.x + threadIdx.x;
    int npix = 12 * nside * nside;

    if (ipix < npix){

        int order = compute_order(nside);
        //int iring, iphi;
        //ring_idx2theta_phi(ipix, iring, iphi, order, nside);
        //int nphi = d_nphi_ring(iring, nside);

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI)*iphi * iphi/nphi;

        c10::complex<T> chirp_b = compute_complex_exp(coef);
        c10::complex<T> chirp_a = conj(chirp_b);

        x_pad[iring * padding + iphi] = f[ipix] * chirp_b;
        y_pad[iring * padding + iphi] = chirp_a;
        y_pad[iring * padding + padding - nphi + iphi] = chirp_a;

    }
}

template <typename T> __global__ void rfft_pre_process_x_pad(c10::complex<T>* x_pad, const T* f, int padding, int nside){

    int ipix = blockIdx.x * blockDim.x + threadIdx.x;
    int npix = 12 * nside * nside;

    if (ipix < npix){

        int order = compute_order(nside);
        //int iring, iphi;
        //ring_idx2theta_phi(ipix, iring, iphi, order, nside);
        //int nphi = d_nphi_ring(iring, nside);

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI)*iphi * iphi/nphi;

        c10::complex<T> chirp_b = compute_complex_exp(coef);

        x_pad[iring * padding + iphi] = f[ipix] * chirp_b;
    }
}

template <typename T> __global__ void rfft_pre_process_y_pad(c10::complex<T>* y_pad, int padding, int nside){

    int ipix = blockIdx.x * blockDim.x + threadIdx.x;
    int npix = 12 * nside * nside;

    if (ipix < npix){

        int order = compute_order(nside);
        //int iring, iphi;
        //ring_idx2theta_phi(ipix, iring, iphi, order, nside);
        //int nphi = d_nphi_ring(iring, nside);

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI)*iphi * iphi/nphi;

        c10::complex<T> chirp_b = compute_complex_exp(coef);
        c10::complex<T> chirp_a = conj(chirp_b);

        y_pad[iring * padding + iphi] = chirp_a;
        y_pad[iring * padding + padding - nphi + iphi] = chirp_a;
    }
}

template <typename T> __global__ void rfft_post_process(c10::complex<T>* x_pad, c10::complex<T>* ftm, int L, int padding, int nside){

    int ipix = blockIdx.x * blockDim.x + threadIdx.x;
    int npix = 12 * nside * nside;

    if (ipix < npix){

        int order = compute_order(nside);
        //int iring, iphi;
        //ring_idx2theta_phi(ipix, iring, iphi, order, nside);
        //int nphi = d_nphi_ring(iring, nside);

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI)*iphi * iphi/nphi;

        c10::complex<T> chirp_b = compute_complex_exp(coef);
        c10::complex<T> result = x_pad[iring * padding + iphi] * chirp_b / padding;

        result = conj(result);

        if (iphi < std::min(L, nphi/2+1)){
            ftm[iring * L + iphi] = result;
        }

    }

}


template <typename T> __global__ void rfft_phase_shift(c10::complex<T>* ftm, int L, int nside) {

    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int iring = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < L && iring < 4*nside-1){
        
        T phi_ring_offset = d_p2phi_ring<T>(iring, 0, nside);

        c10::complex<T> phase_shift = compute_complex_exp(-phi_ring_offset * m);

        ftm[iring * L + m] *= phase_shift;
    }
}



template <typename T> __global__ void irfft_phase_shift(c10::complex<T>* ftm, int L, int nside) {

    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int iring = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < L && iring < 4*nside-1){

        T phi_ring_offset = d_p2phi_ring<T>(iring, 0, nside);

        c10::complex<T> phase_shift = compute_complex_exp(phi_ring_offset * m);

        ftm[iring * L + m] *= phase_shift;
    }
}



template <typename T> __global__ void irfft_pre_process(c10::complex<T>* ftm, c10::complex<T>* x_pad, c10::complex<T>* y_pad,
    int L, int padding, int nside) {

    int ipix = blockIdx.x * blockDim.x + threadIdx.x;

    int npix = 12 * nside * nside;

    if (ipix < npix){

        int order = compute_order(nside);
        //int iring, iphi;
        //ring_idx2theta_phi(ipix, iring, iphi, order, nside);
        //int nphi = d_nphi_ring(iring, nside);

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);
        
        c10::complex<T> fm;

        if (iphi < std::min(nphi/2+1, L)){
            fm= ftm[iring*L + iphi];
        }
        else if (iphi > nphi - std::min(nphi/2+1, L)){ 
            fm= conj(ftm[iring*L + nphi - iphi]);
        }
        else{
            fm = c10::complex<T>(0, 0);
        }

        T coef = static_cast<T>(PI) * iphi * iphi /nphi;
        c10::complex<T> chirp_a = compute_complex_exp(coef);
        c10::complex<T> chirp_b = conj(chirp_a);

        x_pad[iring * padding + iphi] = conj(fm) * chirp_b;

        y_pad[iring * padding + iphi] = chirp_a;

        if (iphi > 0){
            y_pad[iring * padding + padding - nphi + iphi] = chirp_a;
        }

    }

}

template <typename T> __global__ void irfft_pre_process_x_pad(c10::complex<T>* ftm, c10::complex<T>* x_pad, 
    int L, int padding, int nside) {

    int ipix = blockIdx.x * blockDim.x + threadIdx.x;

    int npix = 12 * nside * nside;

    if (ipix < npix){

        int order = compute_order(nside);
        //int iring, iphi;
        //ring_idx2theta_phi(ipix, iring, iphi, order, nside);
        //int nphi = d_nphi_ring(iring, nside);

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        c10::complex<T> fm;

        if (iphi < std::min(nphi/2+1, L)){
            fm = ftm[iring*L + iphi];
        }
        else if (iphi > nphi- std::min(nphi/2+1, L)){ 
            fm = conj(ftm[iring*L + nphi - iphi]);
        }
        else{
            fm = c10::complex<T>(0, 0);
        }

        T coef = static_cast<T>(PI) * iphi * iphi /nphi;
        c10::complex<T> chirp_a = compute_complex_exp(coef);
        c10::complex<T> chirp_b = conj(chirp_a);

        x_pad[iring * padding + iphi] = conj(fm) * chirp_b;

    }

}


template <typename T> __global__ void irfft_pre_process_y_pad(c10::complex<T>* y_pad, int padding, int nside) {

    int ipix = blockIdx.x * blockDim.x + threadIdx.x;

    int npix = 12 * nside * nside;

    if (ipix < npix){

        int order = compute_order(nside);
        //int iring, iphi;
        //ring_idx2theta_phi(ipix, iring, iphi, order, nside);
        //int nphi = d_nphi_ring(iring, nside);

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI) * iphi * iphi /nphi;
        c10::complex<T> chirp_a = compute_complex_exp(coef);
        //c10::complex<T> chirp_b = conj(chirp_a);

        y_pad[iring * padding + iphi] = chirp_a;

        if (iphi > 0){
            y_pad[iring * padding + padding - nphi + iphi] = chirp_a;
        }

    }

}



template <typename T> __global__ void irfft_post_process(
    c10::complex<T>* x_pad,
    T* f,
    int nside,
    int padding) {

    int ipix = blockIdx.x * blockDim.x + threadIdx.x;
    int npix = 12 * nside * nside;

    if (ipix < npix){

        int order = compute_order(nside);
        //int iring, iphi;
        //ring_idx2theta_phi(ipix, iring, iphi, order, nside);
        //int nphi = d_nphi_ring(iring, nside);

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI) * iphi * iphi / nphi;

        c10::complex<T> chirp_b = compute_complex_exp(-coef);
        c10::complex<T> result = x_pad[iring * padding + iphi] * chirp_b / padding;
        f[ipix] = result.real();
    }
}


// CUDA kernel for computing the number of phi samples for each theta ring
__device__ inline int d_nphi_ring(int t, int nside) {
    if (t >= 0 && t < nside - 1) {
        return 4 * (t + 1);
    } else if (t >= nside - 1 && t <= 3 * nside - 1) {
        return 4 * nside;
    } else if (t > 3 * nside - 1 && t <= 4 * nside - 2) {
        return 4 * (4 * nside - t - 1);
    } else {
        return -1; // Error case, handle appropriately in the kernel
    }
}


__device__ inline int d_cumulative_nphi_ring(int t, int nside) {
    if (t >= 0 && t < nside) {
        return 2 * t * (t + 1);
    } else if (t < 3 * nside) {
        int northern_sum = 2 * nside * (nside + 1);
        int equatorial_count = (t - nside) * 4 * nside;
        return northern_sum + equatorial_count;
    } else if (t < 4 * nside) {
        int total_sum = 12 * nside * nside;
        int remaining_rings = 4 * nside - t - 1;
        int remaining_sum = 2 * remaining_rings * (remaining_rings + 1);
        return total_sum - remaining_sum;
    } else {
        return -1; // Error case
    }
}


template<typename T> __device__ inline T d_p2phi_ring(int t, int p, int nside) {

    // Convert index to phi angle for HEALPix
    // t: theta, index of ring
    // p: phi, index within ring

    T shift = 0.5;
    T factor;

    if ((t + 1 >= nside) && (t + 1 <= 3 * nside)) {
        shift *= (t - nside + 2) % 2;
        factor = static_cast<T>(PI) / (2 * nside);
    } else if (t + 1 > 3 * nside) {
        factor = static_cast<T>(PI) / (2 * (4 * nside - t - 1));
    } else {
        factor = static_cast<T>(PI) / (2 * (t + 1));
    }

    return factor * (p + shift);
}



template <typename I> __device__ inline int d_isqrt(I arg){

    int res = int(sqrtf(float(arg) + 0.5));

    if (res*res>arg) {
        --res;
    }
    else if ((res+1)*(res+1)<=arg) {
        ++res;
    }

    return res;
}


template <typename I> __device__ inline void ring_idx2theta_phi(const I pix, I& iring, I& iphi, 
        const int order, const I nside){

    I npface = nside * nside;
    I ncap = (npface - nside) << 1;
    I npix = 12*npface;

    if (pix < ncap){ // North polar cap

        iring = (1 + d_isqrt(1 + 2 * pix)) >> 1;
        iphi = (pix + 1) - 2 * iring * (iring - 1);

    } else if (pix < (npix - ncap)) { // Equatorial Region

        I ip = pix - ncap;
        I tmp = ip >> (order + 2);
        iring = tmp + nside;
        iphi = ip - tmp * 4 * nside + 1;

    } else { // South Polar Cap
        I ip = npix - pix;
        iring = (1 + d_isqrt(2 * ip - 1)) >> 1;
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        iring = 4*nside - iring;
    }

    iring -= 1;
    iphi -= 1;
}

template <typename I> __device__ inline void computer_iring_iphi_nphi(const I pix, I& iring, I& iphi, I& nphi,
        const int order, const I nside){

    I npface = nside * nside;
    I ncap = (npface - nside) << 1;
    I npix = 12*npface;

    if (pix < ncap){ // North polar cap

        iring = (1 + d_isqrt(1 + 2 * pix)) >> 1;
        iphi = pix - 2 * iring * (iring - 1);
        iring -= 1;
        nphi = 4 * (iring + 1);

    } else if (pix < (npix - ncap)) { // Equatorial Region

        I ip = pix - ncap;
        I tmp = ip >> (order + 2);
        iring = tmp + nside - 1;
        iphi = ip - tmp * 4 * nside;

        nphi = 4 * nside;

    } else { // South Polar Cap
        I ip = npix - pix;
        iring = (1 + d_isqrt(2 * ip - 1)) >> 1;
        iphi = 2 * iring * (iring+1) - ip;
        iring = 4*nside - iring - 1;
        nphi = 4 * (4 * nside - iring - 1);
    }
}


template<typename I> __device__ inline int compute_order(I nside) {

    unsigned int res = 0;
    while (nside > 0x00FF) {res |= 8; nside >>= 8;}
    if (nside > 0x000F) {res |= 4; nside >>= 4;}
    if (nside > 0x0003) {res |= 2;nside >>= 2;}
    if (nside > 0x0001) {res |= 1;}
    return res;
}

template<typename T> void rfft_pre_process_kernel_wrapper(c10::complex<T>* x_pad, c10::complex<T>* y_pad, 
    const T* f, int padding, int nside, at::cuda::CUDAStream& stream) {

    int threadsPerBlock = 256;
    int npix= 12 * nside * nside;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    rfft_pre_process<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(x_pad, y_pad, f, padding, nside);
    
}

template<typename T> void rfft_pre_process_x_pad_kernel_wrapper(c10::complex<T>* x_pad, const T* f, int padding, 
    int nside, at::cuda::CUDAStream& stream) {

    int threadsPerBlock = 256;
    int npix= 12 * nside * nside;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    rfft_pre_process_x_pad<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(x_pad, f, padding, nside);
    
}

template<typename T> void rfft_pre_process_y_pad_kernel_wrapper(c10::complex<T>* y_pad, int padding, 
    int nside, at::cuda::CUDAStream& stream) {

    int threadsPerBlock = 256;
    int npix= 12 * nside * nside;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    rfft_pre_process_y_pad<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(y_pad, padding, nside);
    
}

template<typename T> void rfft_post_process_kernel_wrapper(c10::complex<T>* x_pad, c10::complex<T>* ftm, int L, 
    int padding, int nside, at::cuda::CUDAStream& stream) {

    int threadsPerBlock = 256;
    int npix = 12 * nside * nside;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    rfft_post_process<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(x_pad, ftm, L, padding, nside);
}


template<typename T> void rfft_phase_shift_kernel_wrapper(c10::complex<T>* ftm, int L, int nside, at::cuda::CUDAStream& stream) {
    

    dim3 threadsPerBlock(32, 16);
    dim3 blocksPerGrid((L + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (4*nside + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rfft_phase_shift<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(ftm, L, nside);
}

template<typename T> void irfft_phase_shift_kernel_wrapper(c10::complex<T>* ftm, int L, int nside, at::cuda::CUDAStream& stream) {
    

    dim3 threadsPerBlock(32, 16);
    dim3 blocksPerGrid((L + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (4*nside + threadsPerBlock.y - 1) / threadsPerBlock.y);

    irfft_phase_shift<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(ftm, L, nside);
}

template<typename T> void irfft_pre_process_kernel_wrapper(c10::complex<T>* ftm, c10::complex<T>* x_pad, 
    c10::complex<T>* y_pad, int L, int padding, int nside, at::cuda::CUDAStream& stream) {

    int npix = 12 * nside * nside;
    int threadsPerBlock = 256;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    irfft_pre_process<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(ftm, x_pad, y_pad, L, padding, nside);
    
}

template<typename T> void irfft_pre_process_x_pad_kernel_wrapper(c10::complex<T>* ftm, c10::complex<T>* x_pad, 
    int L, int padding, int nside, at::cuda::CUDAStream& stream) {

    int npix = 12 * nside * nside;
    int threadsPerBlock = 256;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    irfft_pre_process_x_pad<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(ftm, x_pad, L, padding, nside);
    
}

template<typename T> void irfft_pre_process_y_pad_kernel_wrapper(c10::complex<T>* y_pad, int padding, 
    int nside, at::cuda::CUDAStream& stream) {

    int npix = 12 * nside * nside;
    int threadsPerBlock = 256;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    irfft_pre_process_y_pad<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(y_pad, padding, nside);
    
}

template<typename T> void irfft_post_process_kernel_wrapper(c10::complex<T>* x_pad, T* f, int nside, 
    int padding, at::cuda::CUDAStream& stream) {

    int npix = 12 * nside * nside;
    int threadsPerBlock = 256;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    irfft_post_process<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(x_pad, f, nside, padding);
    
}


#define DISPATCH_COMPLEX_FLOAT_TYPES(TYPE, NAME, ...)                     \
  [&] {                                                                   \
    const auto& the_type = TYPE;                                          \
    at::ScalarType _st = ::detail::scalar_type(the_type);                 \
    switch (_st) {                                                        \
      case at::ScalarType::ComplexFloat: {                                \
        using scalar_t = float;                                           \
        return __VA_ARGS__();                                             \
      }                                                                   \
      case at::ScalarType::ComplexDouble: {                               \
        using scalar_t = double;                                          \
        return __VA_ARGS__();                                             \
      }                                                                   \
      default:                                                            \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");    \
    }                                                                     \
  }()


void rfft_pre_process_dispatch(torch::Tensor x_pad, torch::Tensor y_pad, torch::Tensor f, int padding, int nside, at::cuda::CUDAStream& stream) {
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "rfft_pre_process", [&] {
        using scalar_t = typename std::conditional<std::is_same<scalar_t, float>::value, float, double>::type;
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto y_pad_ptr = y_pad.data_ptr<c10::complex<scalar_t>>();
        auto f_ptr = f.data_ptr<scalar_t>();

        rfft_pre_process_kernel_wrapper<scalar_t>(x_pad_ptr, y_pad_ptr, f_ptr, padding, nside, stream);
    });
}


void rfft_pre_process_x_pad_dispatch(torch::Tensor x_pad, torch::Tensor f, int padding, int nside, at::cuda::CUDAStream& stream) {
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "rfft_pre_process_x_pad", [&] {
        using scalar_t = typename std::conditional<std::is_same<scalar_t, float>::value, float, double>::type;
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto f_ptr = f.data_ptr<scalar_t>();

        rfft_pre_process_x_pad_kernel_wrapper<scalar_t>(x_pad_ptr, f_ptr, padding, nside, stream);
    });
}

void rfft_pre_process_y_pad_dispatch(torch::Tensor y_pad, int padding, int nside, at::cuda::CUDAStream& stream) {
    DISPATCH_COMPLEX_FLOAT_TYPES(y_pad.scalar_type(), "rfft_pre_process_y_pad", [&] {
        using scalar_t = typename std::conditional<std::is_same<scalar_t, float>::value, float, double>::type;
        auto y_pad_ptr = y_pad.data_ptr<c10::complex<scalar_t>>();
        rfft_pre_process_y_pad_kernel_wrapper<scalar_t>(y_pad_ptr, padding, nside, stream);
    });
}


void rfft_post_process_dispatch(torch::Tensor x_pad, torch::Tensor ftm, int L, int padding, int nside, at::cuda::CUDAStream& stream) {
    DISPATCH_COMPLEX_FLOAT_TYPES(x_pad.scalar_type(), "rfft_post_process", [&] {
        
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        rfft_post_process_kernel_wrapper<scalar_t>(x_pad_ptr, ftm_ptr, L, padding, nside, stream);
    });
}

void rfft_phase_shift_dispatch(torch::Tensor ftm, int L, int nside, at::cuda::CUDAStream& stream) {
    DISPATCH_COMPLEX_FLOAT_TYPES(ftm.scalar_type(), "rfft_phase_shift", [&] {
        
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        rfft_phase_shift_kernel_wrapper<scalar_t>(ftm_ptr, L, nside, stream);
    });
}

void irfft_phase_shift_dispatch(torch::Tensor ftm, int L, int nside, at::cuda::CUDAStream& stream) {
    DISPATCH_COMPLEX_FLOAT_TYPES(ftm.scalar_type(), "irfft_phase_shift", [&] {
        
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        irfft_phase_shift_kernel_wrapper<scalar_t>(ftm_ptr, L, nside, stream);
    });
}

void irfft_pre_process_dispatch(torch::Tensor ftm, torch::Tensor x_pad, torch::Tensor y_pad, int L, int padding, int nside, at::cuda::CUDAStream& stream) {
    DISPATCH_COMPLEX_FLOAT_TYPES(ftm.scalar_type(), "irfft_pre_process", [&] {
        
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto y_pad_ptr = y_pad.data_ptr<c10::complex<scalar_t>>();
        irfft_pre_process_kernel_wrapper<scalar_t>(ftm_ptr, x_pad_ptr, y_pad_ptr, L, padding, nside, stream);
    });
}

void irfft_pre_process_x_pad_dispatch(torch::Tensor ftm, torch::Tensor x_pad, int L, int padding, int nside, at::cuda::CUDAStream& stream) {
    DISPATCH_COMPLEX_FLOAT_TYPES(ftm.scalar_type(), "irfft_pre_process_x_pad", [&] {
        
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();

        irfft_pre_process_x_pad_kernel_wrapper<scalar_t>(ftm_ptr, x_pad_ptr, L, padding, nside, stream);
    });
}

void irfft_pre_process_y_pad_dispatch(torch::Tensor y_pad, int padding, int nside, at::cuda::CUDAStream& stream) {
    DISPATCH_COMPLEX_FLOAT_TYPES(y_pad.scalar_type(), "irfft_pre_process_y_pad", [&] {
        
        auto y_pad_ptr = y_pad.data_ptr<c10::complex<scalar_t>>();
        irfft_pre_process_y_pad_kernel_wrapper<scalar_t>(y_pad_ptr, padding, nside, stream);
    });
}


void irfft_post_process_dispatch(torch::Tensor x_pad, torch::Tensor f, int nside, int padding, at::cuda::CUDAStream& stream) {
    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "irfft_post_process", [&] {
        using scalar_t = typename std::conditional<std::is_same<scalar_t, float>::value, float, double>::type;
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto f_ptr = f.data_ptr<scalar_t>();
        irfft_post_process_kernel_wrapper<scalar_t>(x_pad_ptr, f_ptr, nside, padding, stream);
    });
}


// batched 

template <typename T>
__global__ void rfft_pre_process_x_pad_batch(c10::complex<T>* x_pad, const T* f, const int padding, 
        const int nside, const int order, const size_t m, const size_t n) {
    
    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring (4*nside-1) by padding

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int npix = 12 * nside * nside;
    int nring = 4 * nside -1;


    // Ensure the indices are within bounds
    if (idx < npix && idy < n && idz < m) {
 
        int iring, iphi, nphi;
        computer_iring_iphi_nphi(idx, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI) * iphi * iphi / nphi;
        c10::complex<T> chirp_b = compute_complex_exp(coef);

        int f_idx = (idz * n + idy)* npix + idx;
        int x_pad_idx =  (idz * n + idy)* nring * padding + iring * padding + iphi;

        x_pad[x_pad_idx] = f[f_idx] * chirp_b;

    }
}


template <typename T> __global__ void rfft_post_process_batch(const c10::complex<T>* x_pad, c10::complex<T>* ftm, 
    const int L, const int padding, const int nside, const int order, const size_t m, const size_t n){

    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int npix = 12 * nside * nside;
    int nring = 4 * nside -1;

    if (idx < npix && idy < n && idz < m){

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(idx, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI)*iphi * iphi/nphi;

        c10::complex<T> chirp_b = compute_complex_exp(coef);

        int x_pad_idx =  (idz * n + idy)* nring * padding + iring * padding + iphi;

        c10::complex<T> result = x_pad[x_pad_idx] * chirp_b / padding;

        result = conj(result);

        int ftm_idx =  (idz * n + idy)* nring * L + iring * L + iphi;

        if (iphi < std::min(L, nphi/2+1)){
            ftm[ftm_idx] = result;
        }

    }

}

template <typename T> __global__ void rfft_phase_shift_batch(c10::complex<T>* ftm, const int L, 
    const int nside, const size_t m, const size_t n) {

    // ftm: 4D tensor m by n by nring by L

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iring = blockIdx.y * blockDim.y + threadIdx.y;

    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int nring = 4 * nside -1;

    if (idx < L && iring < nring && idz < m*n){
        
        T phi_ring_offset = d_p2phi_ring<T>(iring, 0, nside);
        c10::complex<T> phase_shift = compute_complex_exp(-phi_ring_offset * idx);

        int ftm_idx =  idz * nring * L + iring * L + idx;
        ftm[ftm_idx] *= phase_shift;

    }
}


template <typename T> __global__ void x_y_pad_conv_batch(c10::complex<T>* x_pad, const c10::complex<T>* y_pad,
        const int padding, const int nside, const size_t m, const size_t n){

    // x_pad: 4D tensor, m by n by nring by padding
    // y_pad: 2D tensor, nring by padding

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iring = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int nring = 4 * nside -1;

    if (idx < padding && iring < nring && idz < m*n){

        int y_pad_idx =  iring * padding + idx;
        int x_pad_idx =  idz * nring * padding + y_pad_idx;

        x_pad[x_pad_idx] *= y_pad[y_pad_idx];

    }

}


template <typename T> __global__ void irfft_phase_shift_batch(c10::complex<T>* ftm, const int L, 
    const int nside, const size_t m, const size_t n) {

    // ftm: 4D tensor m by n by nring by L

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iring = blockIdx.y * blockDim.y + threadIdx.y;

    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int nring = 4 * nside -1;

    if (idx < L && iring < nring && idz < m*n){
        
        T phi_ring_offset = d_p2phi_ring<T>(iring, 0, nside);
        c10::complex<T> phase_shift = compute_complex_exp(phi_ring_offset * idx);

        int ftm_idx =  idz * nring * L + iring * L + idx;

        ftm[ftm_idx] *= phase_shift;

    }
}


template <typename T> __global__ void irfft_pre_process_x_pad_batch(const c10::complex<T>* ftm, c10::complex<T>* x_pad, 
    const int L, const int padding, const int nside, const int order, const size_t m, const size_t n) {

    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int npix = 12 * nside * nside;
    int nring = 4 * nside -1;

    if (idx < npix && idy < n && idz < m){

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(idx, iring, iphi, nphi, order, nside);

        c10::complex<T> fm;

        if (iphi < std::min(nphi/2+1, L)){
            int ftm_idx =  (idz * n + idy)* nring * L + iring * L + iphi;
            fm = ftm[ftm_idx];
        }
        else if (iphi > nphi - std::min(nphi/2+1, L)){ 
            int ftm_idx =  (idz * n + idy)* nring * L + iring * L + nphi - iphi;
            fm = conj(ftm[ftm_idx]);
        }
        else{
            fm = c10::complex<T>(0, 0);
        }

        T coef = static_cast<T>(PI) * iphi * iphi /nphi;
        c10::complex<T> chirp_a = compute_complex_exp(coef);
        c10::complex<T> chirp_b = conj(chirp_a);

        int x_pad_idx =  (idz * n + idy)* nring * padding + iring * padding + iphi;

        x_pad[x_pad_idx] = conj(fm) * chirp_b;

    }

}


template <typename T> __global__ void irfft_post_process_batch(const c10::complex<T>* x_pad, T* f, const int nside, 
    const int order, const int padding, const size_t m, const size_t n) {

    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring (4*nside-1) by padding

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int npix = 12 * nside * nside;
    int nring = 4 * nside -1;

    if (idx < npix && idy < n && idz < m) {

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(idx, iring, iphi, nphi, order, nside);

        T coef = static_cast<T>(PI) * iphi * iphi / nphi;

        c10::complex<T> chirp_b = compute_complex_exp(-coef);

        int x_pad_idx =  (idz * n + idy)* nring * padding + iring * padding + iphi;
        int f_idx = (idz * n + idy)* npix + idx;

        c10::complex<T> result = x_pad[x_pad_idx] * chirp_b / padding;
        f[f_idx] = result.real();
    }
}


template<typename T> void rfft_pre_process_x_pad_batch_kernel_wrapper(c10::complex<T>* x_pad, const T* f, int padding, 
    int nside, int order, const size_t m, const size_t n) {

    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring (4*nside-1) by padding

    int npix = 12 * nside * nside;

    // Calculate block and grid dimensions
    dim3 blockDim(32, 4, 4); // Example 3D block dimensions
    dim3 gridDim((npix + blockDim.x - 1) / blockDim.x,
                  (n + blockDim.y - 1) / blockDim.y,
                  (m + blockDim.z - 1) / blockDim.z);

    rfft_pre_process_x_pad_batch<<<gridDim, blockDim>>>(x_pad, f, padding, nside, order, m, n);
}


template<typename T> void rfft_post_process_batch_kernel_wrapper(c10::complex<T>* x_pad, c10::complex<T>* ftm, int L, 
    int padding, int nside, int order, const size_t m, const size_t n) {

    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int npix = 12 * nside * nside; 

    // Calculate block and grid dimensions
    dim3 blockDim(32, 4, 4); // Example 3D block dimensions
    dim3 gridDim((npix + blockDim.x - 1) / blockDim.x,
                  (n + blockDim.y - 1) / blockDim.y,
                  (m + blockDim.z - 1) / blockDim.z);

    rfft_post_process_batch<<<gridDim, blockDim>>>(x_pad, ftm, L, padding, nside, order, m, n);
}

template<typename T> void rfft_phase_shift_batch_kernel_wrapper(c10::complex<T>* ftm, int L, 
    int nside, const size_t m, const size_t n) {
    
    // ftm: 4D tensor, m by n by nring by L

    int nring = 4*nside-1;

    dim3 blockDim(32, 4, 4);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
                  (nring + blockDim.y - 1) / blockDim.y,
                  (m*n + blockDim.z - 1) / blockDim.z);

    rfft_phase_shift_batch<<<gridDim, blockDim>>>(ftm, L, nside, m, n);
}


template<typename T> void x_y_pad_conv_batch_kernel_wrapper(c10::complex<T>* x_pad, const c10::complex<T>* y_pad,
        int padding, int nside, const size_t m, const size_t n){


    int nring = 4*nside-1;

    dim3 blockDim(32, 1, 16);

    dim3 gridDim((padding + blockDim.x - 1) / blockDim.x,
                  (nring + blockDim.y - 1) / blockDim.y,
                  (m*n + blockDim.z - 1) / blockDim.z);

    x_y_pad_conv_batch<<<gridDim, blockDim>>>(x_pad, y_pad, padding, nside, m, n);
}


template<typename T> void irfft_phase_shift_batch_kernel_wrapper(c10::complex<T>* ftm, int L, 
    int nside, const size_t m, const size_t n) {
    
    // ftm: 4D tensor, m by n by nring by L

    int nring = 4*nside - 1; 

    dim3 blockDim(32, 4, 4);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
                  (nring + blockDim.y - 1) / blockDim.y,
                  (m*n + blockDim.z - 1) / blockDim.z);

    irfft_phase_shift_batch<<<gridDim, blockDim>>>(ftm, L, nside, m, n);
}

template<typename T> void irfft_pre_process_x_pad_batch_kernel_wrapper(c10::complex<T>* ftm, c10::complex<T>* x_pad, int L, 
    int padding, int nside, int order, const size_t m, const size_t n) {

    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int npix = 12 * nside * nside; 

    // Calculate block and grid dimensions
    dim3 blockDim(32, 4, 4); // Example 3D block dimensions
    dim3 gridDim((npix + blockDim.x - 1) / blockDim.x,
                  (n + blockDim.y - 1) / blockDim.y,
                  (m + blockDim.z - 1) / blockDim.z);

    irfft_pre_process_x_pad_batch<<<gridDim, blockDim>>>(ftm, x_pad, L, padding, nside, order, m, n);
}


template<typename T> void irfft_post_process_batch_kernel_wrapper(c10::complex<T>* x_pad, T* f, int nside, 
    int order, int padding, const size_t m, const size_t n) {

    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring (4*nside-1) by padding

    int npix = 12 * nside * nside; 

    // Calculate block and grid dimensions
    dim3 blockDim(32, 4, 4); // Example 3D block dimensions
    dim3 gridDim((npix + blockDim.x - 1) / blockDim.x,
                  (n + blockDim.y - 1) / blockDim.y,
                  (m + blockDim.z - 1) / blockDim.z);

    irfft_post_process_batch<<<gridDim, blockDim>>>(x_pad, f, nside, order, padding, m, n);
}



// Template structure for complex T4 using c10::complex<T>
template <typename T>
struct complex_T4 {
    c10::complex<T> x;
    c10::complex<T> y;
    c10::complex<T> z;
    c10::complex<T> w;
};

template <typename T>
struct T4 {
    T x;
    T y;
    T z;
    T w;
};


template <typename T>
__global__ void rfft_pre_process_x_pad_batch_float4(c10::complex<T>* x_pad, const T* f, const int padding, 
        const int nside, const int order, const size_t n) {
    
    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring (4*nside-1) by padding

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int npix = 12 * nside * nside;
    int nring = 4 * nside -1;

    int ipix = 4 * idx;

    // Ensure the indices are within bounds
    if (ipix < npix && idy < n) {

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        int tmp_x = idy* nring * padding;

        T4<T> f_val = *reinterpret_cast<const T4<T>*>(&f[idy * npix + ipix]);
        T* f_reg = reinterpret_cast<T*>(&f_val);

        for (int i = 0; i < 4; i++, iphi++){

            T coef = static_cast<T>(PI) * iphi * iphi / nphi;
            c10::complex<T> chirp_b = compute_complex_exp(coef);

            int x_pad_idx =  tmp_x + iring * padding + iphi;

            x_pad[x_pad_idx] = chirp_b * f_reg[i];

            if (iphi == nphi){
                iphi = 0;
                iring += 1;
                nphi = d_nphi_ring(iring, nside);
            }
        }

    }
}

template <typename T> __global__ void rfft_post_process_batch_float4(const c10::complex<T>* x_pad, c10::complex<T>* ftm, 
    const int L, const int padding, const int nside, const int order, const size_t n){

    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int npix = 12 * nside * nside;
    int nring = 4 * nside -1;

    int ipix = 4 * idx;

    if (ipix < npix && idy < n){

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        int tmp_x = idy * nring * padding;
        int tmp_ftm = idy * nring * L;

        for (int i = 0; i < 4; i++, iphi++){

            T coef = static_cast<T>(PI)*iphi * iphi/nphi;

            c10::complex<T> chirp_b = compute_complex_exp(coef);
            int x_pad_idx =  tmp_x + iring * padding + iphi;
            c10::complex<T> result = x_pad[x_pad_idx] * chirp_b / padding;

            result = conj(result);
            int ftm_idx =  tmp_ftm + iring * L + iphi;

            if (iphi < std::min(L, nphi/2+1)){
                ftm[ftm_idx] = result;
            }

            if (iphi == nphi){
                iphi = 0;
                iring += 1;
                nphi = d_nphi_ring(iring, nside);
            }
        }
    }
}


template <typename T> __global__ void rfft_phase_shift_batch_float4(c10::complex<T>* ftm, const int L, 
    const int nside, const size_t n) {

    // ftm: 4D tensor m by n by nring by L

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iring = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int nring = 4 * nside -1;

    int iL = idx * 4;

    if (iL < L && iring < nring && idz < n){
        
        T phi_ring_offset = d_p2phi_ring<T>(iring, 0, nside);

        int ftm_idx =  idz * nring * L + iring * L + iL;

        for (int i = 0; i < 4 && iL < L; ++i, ++iL, ++ftm_idx) {
            c10::complex<T> phase_shift = compute_complex_exp(-phi_ring_offset * iL);
            ftm[ftm_idx] *= phase_shift;
        }
    }
}


template <typename T> __global__ void x_y_pad_conv_batch_float4(c10::complex<T>* x_pad, const c10::complex<T>* y_pad,
        const int padding, const int nside, const size_t m, const size_t n){

    // x_pad: 4D tensor, m by n by nring by padding
    // y_pad: 2D tensor, nring by padding

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iring = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int nring = 4 * nside -1;

    int ipad = idx *4;

    if (ipad < padding && iring < nring && idz < m*n){

        int y_pad_idx =  iring * padding + ipad;
        int x_pad_idx =  idz * nring * padding + y_pad_idx;

        // Use reinterpret_cast to treat memory as complex_T4<T>
        const complex_T4<T>* y_val = reinterpret_cast<const complex_T4<T>*>(&y_pad[y_pad_idx]);
        complex_T4<T>* x_val = reinterpret_cast<complex_T4<T>*>(&x_pad[x_pad_idx]);

        // Perform multiplication
        x_val->x *= y_val->x;
        x_val->y *= y_val->y;
        x_val->z *= y_val->z;
        x_val->w *= y_val->w;
    }

}

template <typename T> __global__ void irfft_phase_shift_batch_float4(c10::complex<T>* ftm, const int L, 
    const int nside, const size_t n) {

    // ftm: 4D tensor m by n by nring by L

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iring = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int nring = 4 * nside -1;

    int iL = idx * 4;

    if (iL < L && iring < nring && idz < n){
        
        T phi_ring_offset = d_p2phi_ring<T>(iring, 0, nside);

        int ftm_idx =  idz * nring * L + iring * L + iL;

        for (int i = 0; i < 4 && iL < L; ++i, ++iL, ++ftm_idx) {
            c10::complex<T> phase_shift = compute_complex_exp(phi_ring_offset * iL);
            ftm[ftm_idx] *= phase_shift;
        }
    }
}



template <typename T> __global__ void irfft_pre_process_x_pad_batch_float4(const c10::complex<T>* ftm, c10::complex<T>* x_pad, 
    const int L, const int padding, const int nside, const int order, const size_t n) {

    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int npix = 12 * nside * nside;
    int nring = 4 * nside -1;
    int ipix = 4 * idx;

    if (ipix < npix && idy < n){

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        c10::complex<T> fm;

        const int tmp = idy * nring ;

        for (int i = 0; i < 4; i++, iphi++){

            if (iphi < std::min(nphi/2+1, L)){
                int ftm_idx =  (tmp + iring) * L + iphi;
                fm = ftm[ftm_idx];
            }
            else if (iphi > nphi - std::min(nphi/2+1, L)){ 
                int ftm_idx =  (tmp + iring) * L + nphi - iphi;
                fm = conj(ftm[ftm_idx]);
            }
            else{
                fm = c10::complex<T>(0.0f, 0.0f);
            }

            T coef = static_cast<T>(PI) * iphi * iphi /nphi;
            c10::complex<T> chirp_b = compute_complex_exp(-coef);
            int x_pad_idx =  (tmp + iring) * padding + iphi;
            x_pad[x_pad_idx] = conj(fm) * chirp_b;

            if (iphi == nphi){
                iphi = 0;
                iring += 1;
                nphi = d_nphi_ring(iring, nside);
            }
        }
    }
}


template <typename T> __global__ void irfft_post_process_batch_float4(const c10::complex<T>* x_pad, T* f, const int nside, 
    const int order, const int padding, const size_t n) {

    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring (4*nside-1) by padding

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int npix = 12 * nside * nside;
    int nring = 4 * nside -1;
    int ipix = 4 * idx;

    if (ipix < npix && idy < n) {

        int iring, iphi, nphi;
        computer_iring_iphi_nphi(ipix, iring, iphi, nphi, order, nside);

        const int tmp = idy* nring * padding;

        T4<T>* f_val = reinterpret_cast<T4<T>*>(&f[idy * npix + ipix]);

        for (int i = 0; i  < 4; i++, iphi++){

            T coef = static_cast<T>(PI) * iphi * iphi / nphi;
            c10::complex<T> chirp_b = compute_complex_exp(-coef);

            int x_pad_idx =  tmp + iring * padding + iphi;

            c10::complex<T> result = x_pad[x_pad_idx] * chirp_b;

            reinterpret_cast<T*>(&f_val->x)[i] = result.real() / padding;

            if (iphi == nphi){
                iphi = 0;
                iring += 1;
                nphi = d_nphi_ring(iring, nside);
            }
        }
    }
}


// Kernel wrapper
template<typename T> void rfft_pre_process_x_pad_batch_float4_kernel_wrapper(c10::complex<T>* x_pad, const T* f, int padding, 
    int nside, int order, const size_t n, at::cuda::CUDAStream& stream) {

    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring (4*nside-1) by padding

    int npix = 12 * nside * nside;

    // Calculate block and grid dimensions
    dim3 blockDim(64, 4); // Example 3D block dimensions
    dim3 gridDim((npix/4 + blockDim.x - 1) / blockDim.x,
                  (n + blockDim.y - 1) / blockDim.y);

    rfft_pre_process_x_pad_batch_float4<<<gridDim, blockDim, 0, stream.stream()>>>(x_pad, f, padding, nside, order, n);
}

template<typename T> void rfft_post_process_batch_float4_kernel_wrapper(c10::complex<T>* x_pad, c10::complex<T>* ftm, int L, 
    int padding, int nside, int order, const size_t n, at::cuda::CUDAStream& stream) {

    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int npix = 12 * nside * nside; 

    // Calculate block and grid dimensions
    dim3 blockDim(64, 4); // Example 3D block dimensions
    dim3 gridDim((npix/4 + blockDim.x ) / blockDim.x,
                  (n + blockDim.y - 1) / blockDim.y);

    rfft_post_process_batch_float4<<<gridDim, blockDim, 0, stream.stream()>>>(x_pad, ftm, L, padding, nside, order, n);
}

template<typename T> void rfft_phase_shift_batch_float4_kernel_wrapper(c10::complex<T>* ftm, int L, 
    int nside, const size_t n, at::cuda::CUDAStream& stream) {
    
    // ftm: 4D tensor, m by n by nring by L

    int nring = 4*nside-1;

    dim3 blockDim(32, 4, 4);
    dim3 gridDim((L/4 + blockDim.x) / blockDim.x,
                  (nring + blockDim.y - 1) / blockDim.y,
                  (n + blockDim.z - 1) / blockDim.z);

    rfft_phase_shift_batch_float4<<<gridDim, blockDim, 0, stream.stream()>>>(ftm, L, nside, n);
}


template<typename T> void x_y_pad_conv_batch_float4_kernel_wrapper(c10::complex<T>* x_pad, const c10::complex<T>* y_pad,
        int padding, int nside, const size_t m, const size_t n, at::cuda::CUDAStream& stream){


    int nring = 4*nside-1;

    dim3 blockDim(32, 1, 16);

    dim3 gridDim((padding/4 + blockDim.x - 1) / blockDim.x,
                  (nring + blockDim.y - 1) / blockDim.y,
                  (m*n + blockDim.z - 1) / blockDim.z);

    x_y_pad_conv_batch_float4<<<gridDim, blockDim, 0, stream.stream()>>>(x_pad, y_pad, padding, nside, m, n);
}

template<typename T> void irfft_phase_shift_batch_float4_kernel_wrapper(c10::complex<T>* ftm, int L, 
    int nside, const size_t n, at::cuda::CUDAStream& stream) {
    
    // ftm: 4D tensor, m by n by nring by L

    int nring = 4*nside-1;

    dim3 blockDim(32, 4, 4);
    dim3 gridDim((L/4 + blockDim.x) / blockDim.x,
                  (nring + blockDim.y - 1) / blockDim.y,
                  (n + blockDim.z - 1) / blockDim.z);

    irfft_phase_shift_batch_float4<<<gridDim, blockDim, 0, stream.stream()>>>(ftm, L, nside, n);
}


template<typename T> void irfft_pre_process_x_pad_batch_float4_kernel_wrapper(c10::complex<T>* ftm, c10::complex<T>* x_pad, int L, 
    int padding, int nside, int order, const size_t n, at::cuda::CUDAStream& stream) {

    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int npix = 12 * nside * nside; 

    // Calculate block and grid dimensions
    dim3 blockDim(64, 4); // Example 3D block dimensions
    dim3 gridDim((npix/4 + blockDim.x - 1) / blockDim.x,
                  (n + blockDim.y - 1) / blockDim.y);

    irfft_pre_process_x_pad_batch_float4<<<gridDim, blockDim, 0, stream.stream()>>>(ftm, x_pad, L, padding, nside, order, n);
}


template<typename T> void irfft_post_process_batch_float4_kernel_wrapper(c10::complex<T>* x_pad, T* f, int nside, 
    int order, int padding, const size_t n, at::cuda::CUDAStream& stream) {

    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring (4*nside-1) by padding

    int npix = 12 * nside * nside; 

    // Calculate block and grid dimensions
    dim3 blockDim(64, 4); // Example 3D block dimensions
    dim3 gridDim((npix/4 + blockDim.x - 1) / blockDim.x,
                  (n + blockDim.y - 1) / blockDim.y);

    irfft_post_process_batch_float4<<<gridDim, blockDim, 0, stream.stream()>>>(x_pad, f, nside, order, padding, n);
}


// Dispatch to handle different data type
void rfft_pre_process_x_pad_batch_dispatch(torch::Tensor x_pad, torch::Tensor f, int padding, int nside, int order, at::cuda::CUDAStream& stream) {

    int n = 1;
    for (int i = 0; i < f.dim() - 1; ++i) {
        n *= f.size(i);
    }

    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "rfft_pre_process_x_pad_batch", [&] {
        using scalar_t = typename std::conditional<std::is_same<scalar_t, float>::value, float, double>::type;
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto f_ptr = f.data_ptr<scalar_t>();

        rfft_pre_process_x_pad_batch_float4_kernel_wrapper<scalar_t>(x_pad_ptr, f_ptr, padding, nside, order, n, stream);
    });
}


void rfft_post_process_batch_dispatch(torch::Tensor x_pad, torch::Tensor ftm, int L, int padding, int nside, int order, at::cuda::CUDAStream& stream) {

    int n = 1;
    for (int i = 0; i < ftm.dim() - 2; ++i) {
        n *= ftm.size(i);
    }

    DISPATCH_COMPLEX_FLOAT_TYPES(x_pad.scalar_type(), "rfft_post_process_batch", [&] {
        
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        rfft_post_process_batch_float4_kernel_wrapper<scalar_t>(x_pad_ptr, ftm_ptr, L, padding, nside, order, n, stream);
    });
}

void rfft_phase_shift_batch_dispatch(torch::Tensor ftm, int L, int nside, at::cuda::CUDAStream& stream) {

    int n = 1;
    for (int i = 0; i < ftm.dim() - 2; ++i) {
        n *= ftm.size(i);
    }

    DISPATCH_COMPLEX_FLOAT_TYPES(ftm.scalar_type(), "rfft_phase_shift_batch", [&] {
        
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        rfft_phase_shift_batch_float4_kernel_wrapper<scalar_t>(ftm_ptr, L, nside, n, stream);
    });
}


void x_y_pad_conv_batch_dispatch(torch::Tensor x_pad, torch::Tensor y_pad, int padding, int nside, at::cuda::CUDAStream& stream){

    // x_pad: 4D tensor, m by n by nring by padding
    // y_pad: 2D tensor, nring by padding

    int m = x_pad.size(0);
    int n = x_pad.size(1);

    DISPATCH_COMPLEX_FLOAT_TYPES(x_pad.scalar_type(), "x_y_pad_conv_batch", [&] {
        
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto y_pad_ptr = y_pad.data_ptr<c10::complex<scalar_t>>();

        x_y_pad_conv_batch_float4_kernel_wrapper<scalar_t>(x_pad_ptr, y_pad_ptr, padding, nside, m, n, stream);
    });
}



void irfft_phase_shift_batch_dispatch(torch::Tensor ftm, int L, int nside, at::cuda::CUDAStream& stream) {
    
    int n = 1;
    for (int i = 0; i < ftm.dim() - 2; ++i) {
        n *= ftm.size(i);
    }

    DISPATCH_COMPLEX_FLOAT_TYPES(ftm.scalar_type(), "irfft_phase_shift_batch", [&] {
        
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        irfft_phase_shift_batch_float4_kernel_wrapper<scalar_t>(ftm_ptr, L, nside, n, stream);
    });
}

void irfft_pre_process_x_pad_batch_dispatch(torch::Tensor ftm, torch::Tensor x_pad, int L, int padding, int nside, int order, at::cuda::CUDAStream& stream) {
    
    int n = 1;
    for (int i = 0; i < ftm.dim() - 2; ++i) {
        n *= ftm.size(i);
    }

    DISPATCH_COMPLEX_FLOAT_TYPES(ftm.scalar_type(), "irfft_pre_process_x_pad_batch", [&] {
        
        auto ftm_ptr = ftm.data_ptr<c10::complex<scalar_t>>();
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();

        irfft_pre_process_x_pad_batch_float4_kernel_wrapper<scalar_t>(ftm_ptr, x_pad_ptr, L, padding, nside, order, n, stream);
    });
}

void irfft_post_process_batch_dispatch(torch::Tensor x_pad, torch::Tensor f, int nside, int order, int padding, at::cuda::CUDAStream& stream) {
    
    int n = 1;
    for (int i = 0; i < f.dim() - 1; ++i) {
        n *= f.size(i);
    }

    AT_DISPATCH_FLOATING_TYPES(f.scalar_type(), "irfft_post_process_batch", [&] {
        using scalar_t = typename std::conditional<std::is_same<scalar_t, float>::value, float, double>::type;
        auto x_pad_ptr = x_pad.data_ptr<c10::complex<scalar_t>>();
        auto f_ptr = f.data_ptr<scalar_t>();
        irfft_post_process_batch_float4_kernel_wrapper<scalar_t>(x_pad_ptr, f_ptr, nside, order, padding, n, stream);
    });
}
