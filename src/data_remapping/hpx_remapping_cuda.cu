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
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>


template <typename I> __device__ inline I d_spread_bits(I v);
template <typename I> __device__ inline I d_compress_bits(I v);
template <typename I> __device__ inline int d_isqrt(I arg);
template <typename I> __device__ inline void correct_xy_orient(int& ix, int& iy, const int order, const I nside, const bool flip, const int k);

template<typename I> int compute_order(I nside);

template <typename I>
__global__ void flat2flat_kernel(I* pix_array, const int order, const I nside, const size_t num_elements, const bool flip, const int k){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements){

        I pix = pix_array[tid];

        int face_num = pix >> (2 * order);
        int iy = (pix >> order) & (nside - 1);
        int ix = pix & (nside - 1);

        // Apply the correction to the orientation
        correct_xy_orient(ix, iy, order, nside, flip, k);

        pix_array[tid] = (face_num << (2 * order)) + (iy << order) + ix;

    }
}


template <typename I>
__global__ void xyf2flat_kernel(I* pix_array, const int* ix_array, const int* iy_array, const int* face_num_array,
        const int order, const I nside, const size_t num_elements, const bool flip, const int k){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements){

        int ix = ix_array[tid];
        int iy = iy_array[tid];
        int face_num = face_num_array[tid];

        // Apply the correction to the orientation
        correct_xy_orient(ix, iy, order, nside, flip, k);

        pix_array[tid] = (face_num << (2 * order)) + (iy << order) + ix;

    }
}


template <typename I>
__global__ void flat2xyf_kernel(const I* pix_array, int* ix_array, int* iy_array, int* face_num_array,
        const int order, const I nside, const size_t num_elements, const bool flip, const int k){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements){

        I pix = pix_array[tid];

        int face_num = pix >> (2 * order);
        int iy = (pix >> order) & (nside - 1);
        int ix = pix & (nside - 1);

        // Apply the correction to the orientation
        correct_xy_orient(ix, iy, order, nside, flip, k);

        face_num_array[tid] = face_num;
        iy_array[tid] = iy;
        ix_array[tid] = ix;

    }
}

template <typename I>
__device__ inline void correct_xy_orient(int& ix, int& iy, const int order, const I nside, const bool flip, const int k) {
    int new_ix, new_iy;

    if (flip) {
        int temp = ix;
        ix = iy;
        iy = temp;
    }

    if (k == 1) { // 90 degrees counterclockwise
        new_ix = -iy - 1;
        new_iy = ix;
    } else if (k == 2) { // 180 degrees
        new_ix = -ix - 1;
        new_iy = -iy - 1;
    } else if (k == 3) { // 270 degrees counterclockwise
        new_ix = iy;
        new_iy = -ix - 1;
    } else { // k == 0, no change
        new_ix = ix;
        new_iy = iy;
    }

    new_ix &= (nside - 1);
    if (new_ix < 0) {
        new_ix += nside;
    }
    new_iy &= (nside - 1);
    if (new_iy < 0) {
        new_iy += nside;
    }

    ix = new_ix;
    iy = new_iy;
}

template <typename I, typename T> 
__global__ void rearrange_data_kernel_naive(const T* d_data_in, T* d_data_out, const I* d_pix_array, const size_t num_elements){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements){
        I pix = d_pix_array[tid];
        d_data_out[pix] = d_data_in[tid];
        
    }
}


template <typename I>
 __global__ void initialize_pix_array(I* pix_array, const size_t num_elements){

     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid < num_elements){
        pix_array[tid] = tid;
     }

}


template <typename I> 
__global__ void ring2xyf_kernel(const I* pix_array, int* ix_array, int* iy_array, int* face_num_array,
    const int order, const I nside, const size_t num_elements){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements){

        I npface = nside * nside;
        I ncap = (npface - nside) << 1;
        I npix = 12*npface;

        I pix = pix_array[tid];
        I iring, iphi, kshift, nr;

        int face_num;

        if (pix < ncap){ // North polar cap
            iring = (1 + d_isqrt(1 + 2 * pix)) >> 1;
            iphi = (pix + 1) - 2 * iring * (iring - 1);
            kshift = 0;
            nr = iring;
            face_num = (iphi-1)/nr;
        } else if (pix < (npix - ncap)) { // Equatorial Region
            I ip = pix - ncap;
            I tmp = ip >> (order + 2);
            iring = tmp + nside;
            iphi = ip - tmp * 4 * nside + 1;
            kshift = (iring + nside) & 1;
            nr = nside;

            I ire = tmp + 1;
            I irm = 2*nside + 1 - tmp;
            I ifm = iphi - (ire >> 1) + nside - 1;
            I ifp = iphi - (irm >> 1) + nside - 1;
 
            ifm >>= order;
            ifp >>= order;
     
            face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8));
        } else { // South Polar Cap
            I ip = npix - pix;
            iring = (1 + d_isqrt(2 * ip - 1)) >> 1;
            iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
            kshift = 0;
            nr = iring;
            iring = 4*nside - iring;
            face_num = (iphi-1)/nr + 8;
        }


        I irt = iring - ((2 + (face_num >> 2)) * nside) + 1;
        I ipt = ((face_num & 3) <<1) - ((face_num >> 2) &1) + 1;
        ipt = 2*iphi - ipt*nr - kshift - 1;

        if (ipt >= 2*nside) {
            ipt -= 8 * nside;
        }

        ix_array[tid] = (ipt - irt) >> 1;
        iy_array[tid] = (-ipt - irt) >> 1;
        face_num_array[tid] = face_num;

    }
}


template<typename I> 
__global__ void xyf2ring_kernel(I* pix_array, const int* ix_array, const int* iy_array, const int* face_num_array, 
        const int order, const I nside, const size_t num_elements) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements){
        int ix = ix_array[tid];
        int iy = iy_array[tid];
        int face_num = face_num_array[tid];

        I nl4 = 4 * nside;

        I jr = ((face_num >> 2)+2)*nside -ix -iy -1;

        I nr, kshift, n_before;

        bool shifted;

        I npface = nside * nside;
        I ncap = (npface - nside) << 1;
        I npix = 12*npface;

        if (jr < nside){

            shifted = true;
            nr = 4 *jr;
            n_before = 2 * jr * (jr - 1);

        } else if (jr < 3*nside){

            shifted = ((jr - nside) & 1) == 0;
            nr = 4 * nside;
            n_before = ncap + (jr - nside) * nr;

        } else {

            shifted = true;
            I ring_nr = 4 * nside - jr;
            nr = 4 * ring_nr;
            n_before = npix - 2 * ring_nr * (ring_nr + 1);
        }

        nr >>= 2;
        kshift = 1 - shifted;

        I jp =  ((face_num & 3) <<1) - ((face_num >> 2) &1) + 1;
        jp = (jp * nr + ix - iy + 1 + kshift) / 2;

        if (jp < 1) {
            jp += nl4;
        }

        pix_array[tid] = n_before + jp - 1;
    }
}

template <typename I> 
__global__ void nest2xyf_kernel(const I* pix_array, int* ix_array, int* iy_array, int* face_num_array,
    const int order, const I nside, const size_t num_elements) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements){

        I npface = nside * nside;
        I pix = pix_array[tid];
        int face_num = pix >> (2*order);
        pix &= (npface - 1);

        int ix = d_compress_bits(pix);
        int iy = d_compress_bits(pix>>1);

        ix_array[tid] = ix;
        iy_array[tid] = iy;
        face_num_array[tid] = face_num;
    }
}

template <typename I> 
__global__ void xyf2nest_kernel(I* pix_array, const int* ix_array, const int* iy_array, const int* face_num_array,
    const int order, const I nside, const size_t num_elements){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements){

        int ix = ix_array[tid];
        int iy = iy_array[tid];
        int face_num = face_num_array[tid];

        pix_array[tid] = (I(face_num) << (2 * order)) + d_spread_bits<I>(ix) + (d_spread_bits<I>(iy) << 1);
    }

}

template <typename I> __device__ inline I d_spread_bits(I v);
template <typename I> __device__ inline I d_compress_bits(I v);


template <> __device__ inline int d_spread_bits<int>(int v) {

    int res = v & 0xffff;
    res = (res ^ (res << 8)) & 0x00ff00ff;
    res = (res ^ (res << 4)) & 0x0f0f0f0f;
    res = (res ^ (res << 2)) & 0x33333333;
    res = (res ^ (res << 1)) & 0x55555555;

    return res;
}

template <> __device__ inline int d_compress_bits<int>(int v) {

    int res = v & 0x55555555;
    res = (res ^ (res >> 1)) & 0x33333333;
    res = (res ^ (res >> 2)) & 0x0f0f0f0f;
    res = (res ^ (res >> 4)) & 0x00ff00ff;
    res = (res ^ (res >> 8)) & 0x0000ffff;

    return res;
}


template <> __device__ inline uint64_t d_spread_bits<uint64_t>(uint64_t v) {

    uint64_t res = v & 0xffffffff;
    res = (res^(res<<16)) & 0x0000ffff0000ffff;
    res = (res^(res<< 8)) & 0x00ff00ff00ff00ff;
    res = (res^(res<< 4)) & 0x0f0f0f0f0f0f0f0f;
    res = (res^(res<< 2)) & 0x3333333333333333;
    res = (res^(res<< 1)) & 0x5555555555555555;

    return res;
}


template <> __device__ inline uint64_t d_compress_bits<uint64_t>(uint64_t v) {

    uint64_t res = v & 0x5555555555555555;
    res = (res^(res>> 1)) & 0x3333333333333333;
    res = (res^(res>> 2)) & 0x0f0f0f0f0f0f0f0f;
    res = (res^(res>> 4)) & 0x00ff00ff00ff00ff;
    res = (res^(res>> 8)) & 0x0000ffff0000ffff;
    res = (res^(res>>16)) & 0x00000000ffffffff;

    return res;
}


template <typename I> __device__ inline int d_isqrt(I arg){

    int res = int(sqrt(double(arg) + 0.5));

    if (res*res>arg) {
        --res;
    }
    else if ((res+1)*(res+1)<=arg) {
        ++res;
    }

    return res;
}

template<typename I> int compute_order(I nside) {

    unsigned int res = 0;
    while (nside > 0x00FF) {res |= 8; nside >>= 8;}
    if (nside > 0x000F) {res |= 4; nside >>= 4;}
    if (nside > 0x0003) {res |= 2;nside >>= 2;}
    if (nside > 0x0001) {res |= 1;}
    return res;
}



template <typename T>
void xy2xy_kernel_wrapper(torch::Tensor data_xy_in, torch::Tensor data_xy_out, const int src_origin, const bool src_clockwise,
        const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements){

    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    int order = compute_order(nside);

    int* d_pix_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));

    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) {k += 4;}

    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, order, nside, num_elements, flip, k);
    rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_xy_in.data_ptr<T>(), data_xy_out.data_ptr<T>(), d_pix_array, num_elements);

    cudaFree(d_pix_array);
}


template <typename T>
void nest2xy_kernel_wrapper(torch::Tensor data_in_nest, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
        const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements){

    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    int order = compute_order(nside);

    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));


    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) {k += 4;}


    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    nest2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
    rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_nest.data_ptr<T>(), data_in_xy.data_ptr<T>(), d_pix_array, num_elements);

    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);

}

template <typename T>
void xy2nest_kernel_wrapper(torch::Tensor data_in_xy, torch::Tensor data_in_nest, const int src_origin, const bool src_clockwise,
        const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements){

    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    int order = compute_order(nside);

    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));


    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) {k += 4;}

    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
    xyf2nest_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_xy.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, num_elements);

    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);

}


template <typename T>
void ring2xy_kernel_wrapper(torch::Tensor data_in_ring, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
        const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements){

    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    int order = compute_order(nside);

    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));


    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) {k += 4;}


    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    ring2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
    rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_ring.data_ptr<T>(), data_in_xy.data_ptr<T>(), d_pix_array, num_elements);

    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);

}

template <typename T>
void xy2ring_kernel_wrapper(torch::Tensor data_in_xy, torch::Tensor data_in_ring, const int src_origin, const bool src_clockwise,
        const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements){

    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    int order = compute_order(nside);

    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));


    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) {k += 4;}

    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
    xyf2ring_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_xy.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, num_elements);

    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);

}



// Wrapper function to call the CUDA kernel
template <typename T>
void ring2nest_kernel_wrapper(torch::Tensor data_in_ring, torch::Tensor data_in_nest, const int nside, const size_t num_elements) {

    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    int order = compute_order(nside);

    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    ring2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2nest_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_ring.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, num_elements);
    
    //cudaDeviceSynchronize();

    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}


template <typename T>
void nest2ring_kernel_wrapper(torch::Tensor data_in_nest, torch::Tensor data_in_ring, const int nside, const size_t num_elements) {

    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    int order = compute_order(nside);

    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    nest2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2ring_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_nest.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, num_elements);
    
    //cudaDeviceSynchronize();

    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}


template <typename T>
void benchmark_nest_ring_kernel_wrapper(torch::Tensor data_in_nest, torch::Tensor data_in_ring, const int nside, const size_t num_elements) {

    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    int order = compute_order(nside);

    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    
    float total_elapsed_time_r2n = 0.0f;
    float total_elapsed_time_n2r = 0.0f;

    float total_elapsed_time_n2xy = 0.0f;
    float total_elapsed_time_xy2n = 0.0f;

    float elapsed_time;
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int num_runs = 3;
    int k = 2;
    bool flip = false;

    // warm-up run
    for (int i = 0; i < num_runs; ++i)
    {
        // nest 2 ring
        initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
        nest2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        xyf2ring_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_nest.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, num_elements);

        // ring 2 nest
        initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
        ring2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        xyf2nest_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_ring.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, num_elements);

        // nest2xy
        initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
        nest2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        xyf2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
        rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_nest.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, num_elements);

        // xy2nest
        initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
        flat2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
        xyf2nest_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        rearrange_data_kernel_naive <<<num_blocks, block_size>>>(data_in_ring.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, num_elements);
    }

    cudaDeviceSynchronize();

    num_runs = 10;


    for (int i = 0; i < num_runs; ++i)
    {

        // N2R
        cudaEventRecord(start, stream);

        initialize_pix_array<<<num_blocks, block_size, 0, stream>>>(d_pix_array, num_elements);
        nest2xyf_kernel<<<num_blocks, block_size, 0, stream>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        xyf2ring_kernel<<<num_blocks, block_size, 0, stream>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        rearrange_data_kernel_naive <<<num_blocks, block_size, 0, stream>>>(data_in_nest.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, num_elements);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        elapsed_time = 0.0f;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_elapsed_time_n2r += elapsed_time;

        // R2N

        cudaEventRecord(start, stream);
        
        initialize_pix_array<<<num_blocks, block_size, 0, stream>>>(d_pix_array, num_elements);
        ring2xyf_kernel<<<num_blocks, block_size, 0, stream>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        xyf2nest_kernel<<<num_blocks, block_size, 0, stream>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        rearrange_data_kernel_naive <<<num_blocks, block_size, 0, stream>>>(data_in_ring.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, num_elements);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        elapsed_time = 0.0f;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_elapsed_time_r2n += elapsed_time;

        // nest2xy
        cudaEventRecord(start, stream);

        initialize_pix_array<<<num_blocks, block_size, 0, stream>>>(d_pix_array, num_elements);
        nest2xyf_kernel<<<num_blocks, block_size, 0, stream>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        xyf2flat_kernel<<<num_blocks, block_size, 0, stream>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
        rearrange_data_kernel_naive <<<num_blocks, block_size, 0, stream>>>(data_in_nest.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, num_elements);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        elapsed_time = 0.0f;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_elapsed_time_n2xy += elapsed_time;

        // xy2nest
        cudaEventRecord(start, stream);

        initialize_pix_array<<<num_blocks, block_size, 0, stream>>>(d_pix_array, num_elements);
        flat2xyf_kernel<<<num_blocks, block_size, 0, stream>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
        xyf2nest_kernel<<<num_blocks, block_size, 0, stream>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
        rearrange_data_kernel_naive <<<num_blocks, block_size, 0, stream>>>(data_in_ring.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, num_elements);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        elapsed_time = 0.0f;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_elapsed_time_xy2n += elapsed_time;

    }

    float average_elapsed_time_n2r = total_elapsed_time_n2r / num_runs;
    float average_elapsed_time_r2n = total_elapsed_time_r2n / num_runs;
    float average_elapsed_time_n2xy = total_elapsed_time_n2xy / num_runs;
    float average_elapsed_time_xy2n = total_elapsed_time_xy2n / num_runs;

    
    std::cout << "Average elapsed time used by CUDA data ring2nest kernel over " << num_runs << " runs: " << average_elapsed_time_r2n << "ms" << std::endl;
    std::cout << "Average elapsed time used by CUDA data nest2ring kernel over " << num_runs << " runs: " << average_elapsed_time_n2r << "ms" << std::endl;

    std::cout << "Average elapsed time used by CUDA data nest2XY kernel over " << num_runs << " runs: " << average_elapsed_time_n2xy << "ms" << std::endl;
    std::cout << "Average elapsed time used by CUDA data XY2nest kernel over " << num_runs << " runs: " << average_elapsed_time_xy2n << "ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    cudaDeviceSynchronize();


    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}


// Dispatch functions for each wrapper
void ring2nest_dispatch(torch::Tensor data_in_ring, torch::Tensor data_in_nest, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_ring.scalar_type(), "ring2nest", ([&] {
        ring2nest_kernel_wrapper<scalar_t>(data_in_ring, data_in_nest, nside, num_elements);
    }));
}

void nest2ring_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_ring, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_nest.scalar_type(), "nest2ring", ([&] {
        nest2ring_kernel_wrapper<scalar_t>(data_in_nest, data_in_ring, nside, num_elements);
    }));

}

void nest2xy_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {
    
    AT_DISPATCH_ALL_TYPES(data_in_nest.scalar_type(), "nest2xy", ([&] {
        nest2xy_kernel_wrapper<scalar_t>(data_in_nest, data_in_xy, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));
    
}

void xy2nest_dispatch(torch::Tensor data_in_xy, torch::Tensor data_in_nest, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_xy.scalar_type(), "xy2nest", ([&] {
        xy2nest_kernel_wrapper<scalar_t>(data_in_xy, data_in_nest, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));

}

void ring2xy_dispatch(torch::Tensor data_in_ring, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_ring.scalar_type(), "ring2xy", ([&] {
        ring2xy_kernel_wrapper<scalar_t>(data_in_ring, data_in_xy, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));
    
}

void xy2ring_dispatch(torch::Tensor data_in_xy, torch::Tensor data_in_ring, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_xy.scalar_type(), "xy2ring", ([&] {
        xy2ring_kernel_wrapper<scalar_t>(data_in_xy, data_in_ring, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));
    
}

void xy2xy_dispatch(torch::Tensor data_xy_in, torch::Tensor data_xy_out, const int src_origin, const bool src_clockwise,
                    const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_xy_in.scalar_type(), "xy2xy", ([&] {
        xy2xy_kernel_wrapper<scalar_t>(data_xy_in, data_xy_out, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));
    
}

void benchmark_nest_ring_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_ring, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_nest.scalar_type(), "benchmark_nest_ring", ([&] {
        benchmark_nest_ring_kernel_wrapper<scalar_t>(data_in_nest, data_in_ring, nside, num_elements);
    }));
    
}




template <typename I, typename T>
__global__ void rearrange_data_kernel_3d_batch(const T* d_data_in, T* d_data_out, const I* d_pix_array, 
        const size_t m, const size_t n, const size_t num_elements) {
    
    // d_data_in: m by n by num_elements 3D matrix

    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tid = idz * n * num_elements + idy * num_elements + idx;
    
    if (idz < m && idy < n && idx < num_elements) {

        I pix = d_pix_array[idx];             

        d_data_out[tid-idx + pix] = d_data_in[tid];
    }
}

template <typename I, typename T>
__global__ void rearrange_data_kernel_batch(const T* d_data_in, T* d_data_out, const I* d_pix_array, 
        const size_t n, const size_t num_elements) {

    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int tid = idy * num_elements + idx;
    
    if (idy < n && idx < num_elements) {

        I pix = d_pix_array[idx];             

        d_data_out[tid-idx + pix] = d_data_in[tid];
    }
}

template <typename T>
void xy2xy_batch_kernel_wrapper(torch::Tensor data_xy_in, torch::Tensor data_xy_out, 
                          const int src_origin, const bool src_clockwise,
                          const int dest_origin, const bool dest_clockwise, 
                          const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_xy_in.dim() >= 2, "data_xy_in must be at least a 2D tensor");
    TORCH_CHECK(data_xy_out.dim() >= 2, "data_xy_out must be at least a 2D tensor");

    int n = 1;
    for (int i = 0; i < data_xy_in.dim() - 1; ++i) {
        n *= data_xy_in.size(i);
    }

    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(64, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y);

    // Allocate device memory
    int* d_pix_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, order, nside, num_elements, flip, k);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_batch<<<grid_dim, block_dim>>>(data_xy_in.data_ptr<T>(), data_xy_out.data_ptr<T>(), d_pix_array, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
}


template <typename T>
void nest2xy_batch_kernel_wrapper(torch::Tensor data_in_nest, torch::Tensor data_in_xy, 
                            const int src_origin, const bool src_clockwise,
                            const int dest_origin, const bool dest_clockwise, 
                            const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_nest.dim() >= 2, "data_in_nest must be at least a 2D tensor");
    TORCH_CHECK(data_in_xy.dim() >= 2, "data_in_xy must be at least a 2D tensor");

    int n = 1;
    for (int i = 0; i < data_in_nest.dim() - 1; ++i) {
        n *= data_in_nest.size(i);
    }

    int order = compute_order(nside);

    // Calculate block and grid dimensions
    dim3 block_dim(64, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    nest2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_batch<<<grid_dim, block_dim>>>(data_in_nest.data_ptr<T>(), data_in_xy.data_ptr<T>(), d_pix_array, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}

template <typename T>
void xy2nest_batch_kernel_wrapper(torch::Tensor data_in_xy, torch::Tensor data_in_nest, 
                            const int src_origin, const bool src_clockwise,
                            const int dest_origin, const bool dest_clockwise, 
                            const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_nest.dim() >= 2, "data_in_nest must be at least a 2D tensor");
    TORCH_CHECK(data_in_xy.dim() >= 2, "data_in_xy must be at least a 2D tensor");

    int n = 1;
    for (int i = 0; i < data_in_xy.dim() - 1; ++i) {
        n *= data_in_xy.size(i);
    }

    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(64, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
    xyf2nest_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_batch<<<grid_dim, block_dim>>>(data_in_xy.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}

template <typename T>
void ring2xy_batch_kernel_wrapper(torch::Tensor data_in_ring, torch::Tensor data_in_xy, 
                            const int src_origin, const bool src_clockwise,
                            const int dest_origin, const bool dest_clockwise, 
                            const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_ring.dim() >= 2, "data_in_ring must be at least a 2D tensor");
    TORCH_CHECK(data_in_xy.dim() >= 2, "data_in_xy must be at least a 2D tensor");

    int n = 1;
    for (int i = 0; i < data_in_ring.dim() - 1; ++i) {
        n *= data_in_ring.size(i);
    }

    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(64, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    ring2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_batch<<<grid_dim, block_dim>>>(data_in_ring.data_ptr<T>(), data_in_xy.data_ptr<T>(), d_pix_array, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}


template <typename T>
void xy2ring_batch_kernel_wrapper(torch::Tensor data_in_xy, torch::Tensor data_in_ring, 
                            const int src_origin, const bool src_clockwise,
                            const int dest_origin, const bool dest_clockwise, 
                            const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_ring.dim() >= 2, "data_in_ring must be at least a 2D tensor");
    TORCH_CHECK(data_in_xy.dim() >= 2, "data_in_xy must be at least a 2D tensor");

    int n = 1;
    for (int i = 0; i < data_in_xy.dim() - 1; ++i) {
        n *= data_in_xy.size(i);
    }

    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(64, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
    xyf2ring_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_batch<<<grid_dim, block_dim>>>(data_in_xy.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}


template <typename T>
void ring2nest_batch_kernel_wrapper(torch::Tensor data_in_ring, torch::Tensor data_in_nest, 
                              const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_nest.dim() >= 2, "data_in_nest must be at least a 2D tensor");
    TORCH_CHECK(data_in_ring.dim() >= 2, "data_in_ring must be at least a 2D tensor");

    int n = 1;
    for (int i = 0; i < data_in_ring.dim() - 1; ++i) {
        n *= data_in_ring.size(i);
    }

    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(64, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    ring2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2nest_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_batch<<<grid_dim, block_dim>>>(data_in_ring.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}

template <typename T>
void nest2ring_batch_kernel_wrapper(torch::Tensor data_in_nest, torch::Tensor data_in_ring, 
                              const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_nest.dim() >= 2, "data_in_nest must be at least a 2D tensor");
    TORCH_CHECK(data_in_ring.dim() >= 2, "data_in_ring must be at least a 2D tensor");

    int n = 1;
    for (int i = 0; i < data_in_nest.dim() - 1; ++i) {
        n *= data_in_nest.size(i);
    }

    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(64, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    nest2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2ring_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_batch<<<grid_dim, block_dim>>>(data_in_nest.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}

/*
template <typename T>
void xy2xy_batch_kernel_wrapper(torch::Tensor data_xy_in, torch::Tensor data_xy_out, 
                          const int src_origin, const bool src_clockwise,
                          const int dest_origin, const bool dest_clockwise, 
                          const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_xy_in.dim() == 3, "data_xy_in must be a 3D tensor");
    TORCH_CHECK(data_xy_out.dim() == 3, "data_xy_out must be a 3D tensor");
    
    int m = data_xy_in.size(0); // First dimension
    int n = data_xy_in.size(1); // Second dimension

    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(32, 4, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y,
                  (m + block_dim.z - 1) / block_dim.z);

    // Allocate device memory
    int* d_pix_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, order, nside, num_elements, flip, k);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_3d_batch<<<grid_dim, block_dim>>>(data_xy_in.data_ptr<T>(), data_xy_out.data_ptr<T>(), d_pix_array, m, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
}


template <typename T>
void nest2xy_batch_kernel_wrapper(torch::Tensor data_in_nest, torch::Tensor data_in_xy, 
                            const int src_origin, const bool src_clockwise,
                            const int dest_origin, const bool dest_clockwise, 
                            const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_nest.dim() == 3, "data_in_nest should be a 3D tensor");
    TORCH_CHECK(data_in_xy.dim() == 3, "data_in_xy should be a 3D tensor");
    
    int m = data_in_nest.size(0); // First dimension
    int n = data_in_nest.size(1); // Second dimension

    int order = compute_order(nside);

    // Calculate block and grid dimensions
    dim3 block_dim(32, 4, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y,
                  (m + block_dim.z - 1) / block_dim.z);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    nest2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_3d_batch<<<grid_dim, block_dim>>>(data_in_nest.data_ptr<T>(), data_in_xy.data_ptr<T>(), d_pix_array, m, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}

template <typename T>
void xy2nest_batch_kernel_wrapper(torch::Tensor data_in_xy, torch::Tensor data_in_nest, 
                            const int src_origin, const bool src_clockwise,
                            const int dest_origin, const bool dest_clockwise, 
                            const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_xy.dim() == 3, "data_in_xy must be a 3D tensor");
    TORCH_CHECK(data_in_nest.dim() == 3, "data_in_nest must be a 3D tensor");
    
    int m = data_in_xy.size(0); // First dimension
    int n = data_in_xy.size(1); // Second dimension

    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(32, 4, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y,
                  (m + block_dim.z - 1) / block_dim.z);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
    xyf2nest_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_3d_batch<<<grid_dim, block_dim>>>(data_in_xy.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, m, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}

template <typename T>
void ring2xy_batch_kernel_wrapper(torch::Tensor data_in_ring, torch::Tensor data_in_xy, 
                            const int src_origin, const bool src_clockwise,
                            const int dest_origin, const bool dest_clockwise, 
                            const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_ring.dim() == 3, "data_in_ring must be a 3D tensor");
    TORCH_CHECK(data_in_xy.dim() == 3, "data_in_xy must be a 3D tensor");
    
    int m = data_in_ring.size(0); // First dimension
    int n = data_in_ring.size(1); // Second dimension
    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(32, 4, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y,
                  (m + block_dim.z - 1) / block_dim.z);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    ring2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2flat_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_3d_batch<<<grid_dim, block_dim>>>(data_in_ring.data_ptr<T>(), data_in_xy.data_ptr<T>(), d_pix_array, m, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}


template <typename T>
void xy2ring_batch_kernel_wrapper(torch::Tensor data_in_xy, torch::Tensor data_in_ring, 
                            const int src_origin, const bool src_clockwise,
                            const int dest_origin, const bool dest_clockwise, 
                            const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_xy.dim() == 3, "data_in_xy must be a 3D tensor");
    TORCH_CHECK(data_in_ring.dim() == 3, "data_in_ring must be a 3D tensor");
    
    int m = data_in_xy.size(0); // First dimension
    int n = data_in_xy.size(1); // Second dimension
    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(32, 4, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y,
                  (m + block_dim.z - 1) / block_dim.z);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Compute rotation parameters
    bool flip = src_clockwise != dest_clockwise;
    int rotations = dest_origin - src_origin;
    int k = (dest_clockwise ? -rotations : rotations) & 3;
    if (k < 0) { k += 4; }

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    flat2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements, flip, k);
    xyf2ring_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_3d_batch<<<grid_dim, block_dim>>>(data_in_xy.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, m, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}


template <typename T>
void ring2nest_batch_kernel_wrapper(torch::Tensor data_in_ring, torch::Tensor data_in_nest, 
                              const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_ring.dim() == 3, "data_in_ring must be a 3D tensor");
    TORCH_CHECK(data_in_nest.dim() == 3, "data_in_nest must be a 3D tensor");
    
    int m = data_in_ring.size(0); // First dimension
    int n = data_in_ring.size(1); // Second dimension
    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(32, 4, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y,
                  (m + block_dim.z - 1) / block_dim.z);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    ring2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2nest_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_3d_batch<<<grid_dim, block_dim>>>(data_in_ring.data_ptr<T>(), data_in_nest.data_ptr<T>(), d_pix_array, m, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}

template <typename T>
void nest2ring_batch_kernel_wrapper(torch::Tensor data_in_nest, torch::Tensor data_in_ring, 
                              const int nside, const size_t num_elements) {

    // Check tensor dimensions and sizes
    TORCH_CHECK(data_in_nest.dim() == 3, "data_in_nest must be a 3D tensor");
    TORCH_CHECK(data_in_ring.dim() == 3, "data_in_ring must be a 3D tensor");

    int m = data_in_nest.size(0); // First dimension
    int n = data_in_nest.size(1); // Second dimension
    int order = compute_order(nside);

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1)/ block_size;

    // Calculate block and grid dimensions
    dim3 block_dim(32, 4, 4); // Example 3D block dimensions
    dim3 grid_dim((num_elements + block_dim.x - 1) / block_dim.x,
                  (n + block_dim.y - 1) / block_dim.y,
                  (m + block_dim.z - 1) / block_dim.z);

    // Allocate device memory
    int* d_pix_array;
    int* d_ix_array;
    int* d_iy_array;
    int* d_face_num_array;

    cudaMalloc(&d_pix_array, num_elements * sizeof(int));
    cudaMalloc(&d_ix_array, num_elements * sizeof(int));
    cudaMalloc(&d_iy_array, num_elements * sizeof(int));
    cudaMalloc(&d_face_num_array, num_elements * sizeof(int));

    // Launch kernels
    initialize_pix_array<<<num_blocks, block_size>>>(d_pix_array, num_elements);
    nest2xyf_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);
    xyf2ring_kernel<<<num_blocks, block_size>>>(d_pix_array, d_ix_array, d_iy_array, d_face_num_array, order, nside, num_elements);

    // Adjust kernel launch for 3D data
    rearrange_data_kernel_3d_batch<<<grid_dim, block_dim>>>(data_in_nest.data_ptr<T>(), data_in_ring.data_ptr<T>(), d_pix_array, m, n, num_elements);

    // Free device memory
    cudaFree(d_pix_array);
    cudaFree(d_ix_array);
    cudaFree(d_iy_array);
    cudaFree(d_face_num_array);
}
*/

// Dispatch functions for each wrapper
void ring2nest_batch_dispatch(torch::Tensor data_in_ring, torch::Tensor data_in_nest, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_ring.scalar_type(), "ring2nest_batch", ([&] {
        ring2nest_batch_kernel_wrapper<scalar_t>(data_in_ring, data_in_nest, nside, num_elements);
    }));
}

void nest2ring_batch_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_ring, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_nest.scalar_type(), "nest2ring_batch", ([&] {
        nest2ring_batch_kernel_wrapper<scalar_t>(data_in_nest, data_in_ring, nside, num_elements);
    }));

}

void nest2xy_batch_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {
    
    AT_DISPATCH_ALL_TYPES(data_in_nest.scalar_type(), "nest2xy_batch", ([&] {
        nest2xy_batch_kernel_wrapper<scalar_t>(data_in_nest, data_in_xy, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));
    
}

void xy2nest_batch_dispatch(torch::Tensor data_in_xy, torch::Tensor data_in_nest, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_xy.scalar_type(), "xy2nest_batch", ([&] {
        xy2nest_batch_kernel_wrapper<scalar_t>(data_in_xy, data_in_nest, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));

}

void ring2xy_batch_dispatch(torch::Tensor data_in_ring, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_ring.scalar_type(), "ring2xy_batch", ([&] {
        ring2xy_batch_kernel_wrapper<scalar_t>(data_in_ring, data_in_xy, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));
    
}

void xy2ring_batch_dispatch(torch::Tensor data_in_xy, torch::Tensor data_in_ring, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_in_xy.scalar_type(), "xy2ring_batch", ([&] {
        xy2ring_batch_kernel_wrapper<scalar_t>(data_in_xy, data_in_ring, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));
    
}

void xy2xy_batch_dispatch(torch::Tensor data_xy_in, torch::Tensor data_xy_out, const int src_origin, const bool src_clockwise,
                    const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements) {

    AT_DISPATCH_ALL_TYPES(data_xy_in.scalar_type(), "xy2xy_batch", ([&] {
        xy2xy_batch_kernel_wrapper<scalar_t>(data_xy_in, data_xy_out, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);
    }));
    
}
