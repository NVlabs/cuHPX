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

#ifndef HPX_FFT_H
#define HPX_FFT_H

#include <torch/torch.h>
#include <torch/fft.h>
#include <ATen/ATen.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <complex>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void rfft_pre_process_dispatch(torch::Tensor x_pad, torch::Tensor y_pad, torch::Tensor f, int padding, int nside, at::cuda::CUDAStream& stream);

void rfft_pre_process_x_pad_dispatch(torch::Tensor x_pad, torch::Tensor f, int padding, int nside, at::cuda::CUDAStream& stream);

void rfft_pre_process_y_pad_dispatch(torch::Tensor y_pad, int padding, int nside, at::cuda::CUDAStream& stream);

void rfft_post_process_dispatch(torch::Tensor x_pad, torch::Tensor ftm, int L, int padding, int nside, at::cuda::CUDAStream& stream);

void rfft_phase_shift_dispatch(torch::Tensor ftm, int L, int nside, at::cuda::CUDAStream& stream);

void irfft_phase_shift_dispatch(torch::Tensor ftm, int L, int nside, at::cuda::CUDAStream& stream);

void irfft_pre_process_dispatch(torch::Tensor ftm, torch::Tensor x_pad, torch::Tensor y_pad, int L, int padding, int nside, at::cuda::CUDAStream& stream);

void irfft_pre_process_x_pad_dispatch(torch::Tensor ftm, torch::Tensor x_pad, int L, int padding, int nside, at::cuda::CUDAStream& stream);

void irfft_pre_process_y_pad_dispatch(torch::Tensor y_pad, int padding, int nside, at::cuda::CUDAStream& stream);

void irfft_post_process_dispatch(torch::Tensor x_pad, torch::Tensor f, int nside, int padding, at::cuda::CUDAStream& stream);


void rfft_pre_process_x_pad_batch_dispatch(torch::Tensor x_pad, torch::Tensor f, int padding, int nside, int order, at::cuda::CUDAStream& stream);

void rfft_post_process_batch_dispatch(torch::Tensor x_pad, torch::Tensor ftm, int L, int padding, int nside, int order, at::cuda::CUDAStream& stream);

void rfft_phase_shift_batch_dispatch(torch::Tensor ftm, int L, int nside, at::cuda::CUDAStream& stream);

void irfft_phase_shift_batch_dispatch(torch::Tensor ftm, int L, int nside, at::cuda::CUDAStream& stream);

void irfft_pre_process_x_pad_batch_dispatch(torch::Tensor ftm, torch::Tensor x_pad, int L, int padding, int nside, int order, at::cuda::CUDAStream& stream);

void irfft_post_process_batch_dispatch(torch::Tensor x_pad, torch::Tensor f, int nside, int order, int padding, at::cuda::CUDAStream& stream);

void x_y_pad_conv_batch_dispatch(torch::Tensor x_pad, torch::Tensor y_pad, int padding, int nside, at::cuda::CUDAStream& stream);

template<typename I> inline int compute_order(I nside);

void rfft_pre_process_x_pad_batch_float4_dispatch(torch::Tensor x_pad, torch::Tensor f, int padding, int nside, int order, at::cuda::CUDAStream& stream);

class HealpixFFT{
public:
    // Constructor initializes only the essential variables and FFT plan
    HealpixFFT(int ntheta, int n, int padding, int L, torch::Dtype dtype, torch::Device device, const at::cuda::CUDAStream &stream)
        : ntheta_(ntheta), n_(n), padding_(padding), L_(L), dtype_(dtype), device_(device), y_pad_initialized_(false), stream_(stream){

        if (dtype == torch::kComplexDouble) {
            checkCuFFTError(cufftPlan1d(&plan_, padding_, CUFFT_Z2Z, ntheta_ * n));
        } else if (dtype == torch::kComplexFloat) {
            checkCuFFTError(cufftPlan1d(&plan_, padding_, CUFFT_C2C, ntheta_ * n));
        } else {
            throw std::runtime_error("Unsupported data type for FFT plan.");
        }

        // Set the CUDA stream for the FFT plan
        checkCuFFTError(cufftSetStream(plan_, stream_.stream()));

        // Allocate memory for y_pad tensor
        y_pad_ = torch::zeros({ntheta, padding}, torch::dtype(dtype).device(device));

        // Pre-allocate workspace buffers for CUDA graph compatibility
        x_pad_ = torch::empty({ntheta * n, padding}, torch::dtype(dtype).device(device));
        ftm_ = torch::empty({ntheta * n, L}, torch::dtype(dtype).device(device));
    }

    ~HealpixFFT() {
        checkCuFFTError(cufftDestroy(plan_));
    }

    void initializeYpad(int nside) {
        if (!y_pad_initialized_) {
            rfft_pre_process_y_pad_dispatch(y_pad_, padding_, nside, stream_);
            //execute_forward(y_pad_);
            y_pad_ = torch::fft::fft(y_pad_, y_pad_.size(-1));
            y_pad_initialized_ = true;
        }
    }

    void execute_forward(torch::Tensor& pad) {
        if (dtype_ == torch::kComplexDouble) {
            checkCuFFTError(cufftExecZ2Z(plan_, reinterpret_cast<cufftDoubleComplex*>(pad.data_ptr<c10::complex<double>>()),
                reinterpret_cast<cufftDoubleComplex*>(pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));
        } else if (dtype_ == torch::kComplexFloat) {
            checkCuFFTError(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(pad.data_ptr<c10::complex<float>>()),
                reinterpret_cast<cufftComplex*>(pad.data_ptr<c10::complex<float>>()), CUFFT_FORWARD));
        }
    }

    void execute_inverse(torch::Tensor& pad) {
        if (dtype_ == torch::kComplexDouble) {
            checkCuFFTError(cufftExecZ2Z(plan_, reinterpret_cast<cufftDoubleComplex*>(pad.data_ptr<c10::complex<double>>()),
                reinterpret_cast<cufftDoubleComplex*>(pad.data_ptr<c10::complex<double>>()), CUFFT_INVERSE));
        } else if (dtype_ == torch::kComplexFloat) {
            checkCuFFTError(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(pad.data_ptr<c10::complex<float>>()),
                reinterpret_cast<cufftComplex*>(pad.data_ptr<c10::complex<float>>()), CUFFT_INVERSE));
        }
    }

    const torch::Tensor& getYpad() const {
        return y_pad_;
    }

    // Method to check if reconfiguration is needed
    // Only reconfigure when batch grows larger (n > n_) or other config changes
    bool needsReconfiguration(int ntheta, int n, int padding, int L, torch::Dtype dtype, torch::Device device) const {
        return ntheta_ != ntheta || n_ < n || padding_ != padding || L_ != L || dtype_ != dtype || device_ != device;
    }

    // Method to check and update the stream (non-const)
    void updateStreamIfNeeded(at::cuda::CUDAStream stream) {

        // Set the plan to the new stream
        checkCuFFTError(cufftSetStream(plan_, stream.stream()));
        stream_ = stream;
    }


    // Getters for current configuration
    int getNtheta() const { return ntheta_; }
    int getPadding() const { return padding_; }
    int getL() const { return L_; }
    int getN() const { return n_; }
    torch::Dtype getDtype() const { return dtype_; }
    torch::Device getDevice() const { return device_; }

    // Accessors for pre-allocated buffers
    torch::Tensor& getXpad() { return x_pad_; }
    torch::Tensor& getFtm() { return ftm_; }

private:
    cufftHandle plan_;
    int ntheta_;
    int n_;
    int padding_;
    int L_;
    torch::Dtype dtype_;
    torch::Device device_;
    torch::Tensor y_pad_;
    torch::Tensor x_pad_;
    torch::Tensor ftm_;
    bool y_pad_initialized_;
    at::cuda::CUDAStream stream_;

    void checkCuFFTError(cufftResult result) {
        if (result != CUFFT_SUCCESS) {
            throw std::runtime_error("CUFFT error: " + std::to_string(result));
        }
    }
};


class HealpixIFFT {
public:
    HealpixIFFT(int ntheta, int n, int padding, int nside, torch::Dtype dtype, torch::Device device, at::cuda::CUDAStream stream)
        : ntheta_(ntheta), n_(n), padding_(padding), nside_(nside), dtype_(dtype), device_(device), y_pad_initialized_(false), stream_(stream){

        if (dtype == torch::kComplexDouble) {
            checkCuFFTError(cufftPlan1d(&plan_, padding_, CUFFT_Z2Z, ntheta_ * n));
        } else if (dtype == torch::kComplexFloat) {
            checkCuFFTError(cufftPlan1d(&plan_, padding_, CUFFT_C2C, ntheta_ * n));
        } else {
            throw std::runtime_error("Unsupported data type for FFT plan.");
        }

        // Set the CUDA stream for the FFT plan
        checkCuFFTError(cufftSetStream(plan_, stream_.stream()));

        y_pad_ = torch::zeros({ntheta, padding}, torch::dtype(dtype).device(device));

        // Pre-allocate workspace buffers for CUDA graph compatibility
        x_pad_ = torch::empty({ntheta * n, padding}, torch::dtype(dtype).device(device));

        // Output tensor type is real (not complex)
        auto ftype = dtype == torch::kComplexDouble ? torch::kDouble : torch::kFloat;
        int npix = 12 * nside * nside;
        f_ = torch::empty({n, npix}, torch::dtype(ftype).device(device));
    }

    ~HealpixIFFT() {
        checkCuFFTError(cufftDestroy(plan_));
    }

    void initializeYpad(int nside) {
        if (!y_pad_initialized_) {
            irfft_pre_process_y_pad_dispatch(y_pad_, padding_, nside, stream_);
            y_pad_ = torch::fft::fft(y_pad_, y_pad_.size(-1));
            y_pad_initialized_ = true;
        }
    }

    void execute_forward(torch::Tensor& pad) {
        if (dtype_ == torch::kComplexDouble) {
            checkCuFFTError(cufftExecZ2Z(plan_, reinterpret_cast<cufftDoubleComplex*>(pad.data_ptr<c10::complex<double>>()),
                reinterpret_cast<cufftDoubleComplex*>(pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));
        } else if (dtype_ == torch::kComplexFloat) {
            checkCuFFTError(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(pad.data_ptr<c10::complex<float>>()),
                reinterpret_cast<cufftComplex*>(pad.data_ptr<c10::complex<float>>()), CUFFT_FORWARD));
        }
    }

    void execute_inverse(torch::Tensor& pad) {
        if (dtype_ == torch::kComplexDouble) {
            checkCuFFTError(cufftExecZ2Z(plan_, reinterpret_cast<cufftDoubleComplex*>(pad.data_ptr<c10::complex<double>>()),
                reinterpret_cast<cufftDoubleComplex*>(pad.data_ptr<c10::complex<double>>()), CUFFT_INVERSE));
        } else if (dtype_ == torch::kComplexFloat) {
            checkCuFFTError(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(pad.data_ptr<c10::complex<float>>()),
                reinterpret_cast<cufftComplex*>(pad.data_ptr<c10::complex<float>>()), CUFFT_INVERSE));
        }
    }

    const torch::Tensor& getYpad() const {
        return y_pad_;
    }

    // Method to check if reconfiguration is needed
    // Only reconfigure when batch grows larger (n > n_) or other config changes
    bool needsReconfiguration(int ntheta, int n, int padding, int nside, torch::Dtype dtype, torch::Device device) const {
        return ntheta_ != ntheta || n_ < n || padding_ != padding || nside_ != nside || dtype_ != dtype || device_ != device;
    }

    // Method to check and update the stream (non-const)
    void updateStreamIfNeeded(at::cuda::CUDAStream stream) {
        // Set the plan to the new stream
        checkCuFFTError(cufftSetStream(plan_, stream.stream()));
        stream_ = stream;
    }

    // Getters for current configuration
    int getNtheta() const { return ntheta_; }
    int getPadding() const { return padding_; }
    int getNside() const { return nside_; }
    int getN() const { return n_; }
    torch::Dtype getDtype() const { return dtype_; }
    torch::Device getDevice() const { return device_; }

    // Accessors for pre-allocated buffers
    torch::Tensor& getXpad() { return x_pad_; }
    torch::Tensor& getF() { return f_; }

private:
    cufftHandle plan_;
    int ntheta_;
    int n_;
    int padding_;
    int nside_;
    torch::Dtype dtype_;
    torch::Device device_;
    torch::Tensor y_pad_;
    torch::Tensor x_pad_;
    torch::Tensor f_;
    bool y_pad_initialized_;
    at::cuda::CUDAStream stream_;

    void checkCuFFTError(cufftResult result) {
        if (result != CUFFT_SUCCESS) {
            throw std::runtime_error("CUFFT error: " + std::to_string(result));
        }
    }
};



#endif // HPX_FFT_H
