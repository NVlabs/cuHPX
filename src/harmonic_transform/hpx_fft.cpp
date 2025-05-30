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

#include "hpx_fft.h"
#include <cmath>

torch::Tensor healpix_rfft_batch(torch::Tensor f, int L, int nside) {

    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    // Retrieve the current CUDA stream
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAStreamGuard guard(stream);

    int ndims = f.dim();
    TORCH_CHECK(ndims >= 2, "f must have at least 2 dimensions");

    if (!f.is_contiguous()) {
        f = f.contiguous();
    }

    std::vector<int64_t> batch_dims(f.sizes().begin(), f.sizes().end() - 1);
    int n = std::accumulate(batch_dims.begin(), batch_dims.end(), 1, std::multiplies<int64_t>());

    // Configuration parameters
    int ntheta = 4 * nside - 1;
    int padding = 8 * nside;
    auto device = f.device();
    auto dtype = f.scalar_type() == torch::kDouble ? torch::kComplexDouble : torch::kComplexFloat;
    int order = compute_order(nside);

    // Create FFT object and initialize y_pad if not already done
    static HealpixFFT* fft = nullptr;
    if (!fft || fft->needsReconfiguration(ntheta, n, padding, dtype, device)) {

        delete fft; // Properly deallocate existing object
        fft = new HealpixFFT(ntheta, n, padding, dtype, device, stream);
    } else {
        // If reconfiguration is not needed, ensure the stream is up-to-date
        fft->updateStreamIfNeeded(stream);
    }
    fft->initializeYpad(nside);

    auto ftm_size = batch_dims;
    ftm_size.push_back(ntheta);
    ftm_size.push_back(L);

    auto x_pad_size = batch_dims;
    x_pad_size.push_back(ntheta);
    x_pad_size.push_back(padding);

    auto ftm = torch::zeros(ftm_size, torch::dtype(dtype).device(device));
    auto x_pad = torch::zeros(x_pad_size, torch::dtype(dtype).device(device));

    rfft_pre_process_x_pad_batch_dispatch(x_pad, f, padding, nside, order, stream);

    fft->execute_forward(x_pad);

    x_pad = x_pad * fft->getYpad();

    fft->execute_inverse(x_pad);

    rfft_post_process_batch_dispatch(x_pad, ftm, L, padding, nside, order, stream);
    rfft_phase_shift_batch_dispatch(ftm, L, nside, stream);


    return ftm;
}

torch::Tensor healpix_irfft_batch(torch::Tensor ftm, int L, int nside) {

    // f: 3D tensor, m by n by npix
    // x_pad: 4D tensor, m by n by nring by padding
    // ftm: 4D tensor, m by n by nring by L

    int ndims = ftm.dim();
    TORCH_CHECK(ftm.dim() >= 3, "ftm must be a 3D tensor");

    if (!ftm.is_contiguous()) {
        ftm = ftm.contiguous();
    }

    std::vector<int64_t> batch_dims(ftm.sizes().begin(), ftm.sizes().end() - 2);
    int n = std::accumulate(batch_dims.begin(), batch_dims.end(), 1, std::multiplies<int64_t>());

    // Retrieve the current CUDA stream
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAStreamGuard guard(stream);

    int ntheta = 4 * nside -1;
    int padding = 8 * nside;
    int order = compute_order(nside);

    auto device = ftm.device();
    auto dtype = ftm.scalar_type() == torch::kComplexDouble ? torch::kComplexDouble : torch::kComplexFloat;
    auto ftype = dtype == torch::kComplexDouble ? torch::kDouble : torch::kFloat;

    auto f_size = batch_dims;
    f_size.push_back(12 * nside * nside);

    auto x_pad_size = batch_dims;
    x_pad_size.push_back(ntheta);
    x_pad_size.push_back(padding);

    auto f = torch::zeros(f_size, torch::dtype(ftype).device(device));
    auto x_pad = torch::zeros(x_pad_size, torch::dtype(dtype).device(device));

    // Instantiate FFT object
    static HealpixIFFT* ifft = nullptr;
    if (!ifft || ifft->needsReconfiguration(ntheta, n, padding, dtype, device)) {
        delete ifft; // Properly deallocate existing object
        ifft = new HealpixIFFT(ntheta, n, padding, dtype, device, stream);
    } else {
        // If reconfiguration is not needed, ensure the stream is up-to-date
        ifft->updateStreamIfNeeded(stream);
    }
    ifft->initializeYpad(nside);

    irfft_phase_shift_batch_dispatch(ftm, L, nside, stream);
    irfft_pre_process_x_pad_batch_dispatch(ftm, x_pad, L, padding, nside, order, stream);

    ifft->execute_forward(x_pad);

    x_pad = x_pad * ifft->getYpad();
    //x_y_pad_conv_batch_dispatch(x_pad, ifft->getYpad(), padding, nside);

    ifft->execute_inverse(x_pad);

    irfft_post_process_batch_dispatch(x_pad, f, nside, order, padding, stream);

    return f;

}


torch::Tensor healpix_rfft_class(torch::Tensor f, int L, int nside) {
    // Configuration parameters
    int ntheta = 4 * nside - 1;
    int padding = 8 * nside;
    auto device = f.device();
    auto dtype = f.scalar_type() == torch::kDouble ? torch::kComplexDouble : torch::kComplexFloat;

    // Retrieve the current CUDA stream
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // Use CUDAStreamGuard to set the current stream (though it's already set, this makes it explicit)
    at::cuda::CUDAStreamGuard guard(stream);

    // Create FFT object and initialize y_pad if not already done

    static HealpixFFT* fft = nullptr;
    if (!fft || fft->needsReconfiguration(ntheta, 1, padding, dtype, device)) {
        delete fft; // Properly deallocate existing object
        fft = new HealpixFFT(ntheta, 1, padding, dtype, device, stream);
    } else {
        // If reconfiguration is not needed, ensure the stream is up-to-date
        fft->updateStreamIfNeeded(stream);
    }
    fft->initializeYpad(nside);

    // Allocate tensors and perform FFT operations
    auto ftm = torch::zeros({ntheta, L}, torch::dtype(dtype).device(device));
    auto x_pad = torch::zeros({ntheta, padding}, torch::dtype(dtype).device(device));

    rfft_pre_process_x_pad_dispatch(x_pad, f, padding, nside, stream);
    fft->execute_forward(x_pad);

    x_pad.mul_(fft->getYpad());

    fft->execute_inverse(x_pad);

    // x_pad.div_(padding);

    rfft_post_process_dispatch(x_pad, ftm, L, padding, nside, stream);
    rfft_phase_shift_dispatch(ftm, L, nside, stream);

    return ftm;
}

torch::Tensor healpix_irfft_class(torch::Tensor ftm, int L, int nside) {

    int ntheta = 4 * nside - 1;
    int padding = 8 * nside;

    auto device = ftm.device();
    auto dtype = ftm.scalar_type() == torch::kComplexDouble ? torch::kComplexDouble : torch::kComplexFloat;
    auto ftype = dtype == torch::kComplexDouble ? torch::kDouble : torch::kFloat;

    // Retrieve the current CUDA stream
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // Use CUDAStreamGuard to set the current stream (though it's already set, this makes it explicit)
    at::cuda::CUDAStreamGuard guard(stream);


    auto f = torch::zeros({12 * nside * nside}, torch::dtype(ftype).device(device));
    auto x_pad = torch::zeros({ntheta, padding}, torch::dtype(dtype).device(device));

    // Instantiate FFT object

    static HealpixIFFT* ifft = nullptr;
    if (!ifft || ifft->needsReconfiguration(ntheta, 1, padding, dtype, device)) {
        delete ifft; // Properly deallocate existing object
        ifft = new HealpixIFFT(ntheta, 1, padding, dtype, device, stream);
    } else {
        // If reconfiguration is not needed, ensure the stream is up-to-date
        ifft->updateStreamIfNeeded(stream);
    }
    ifft->initializeYpad(nside);

    irfft_phase_shift_dispatch(ftm, L, nside, stream);

    irfft_pre_process_x_pad_dispatch(ftm, x_pad, L, padding, nside, stream);

    ifft->execute_forward(x_pad);

    x_pad.mul_(ifft->getYpad());

    ifft->execute_inverse(x_pad);

    irfft_post_process_dispatch(x_pad, f, nside, padding, stream);

    return f;
}

// Helper function to check cuFFT errors
void checkCuFFTError(cufftResult result) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error: " << result << std::endl;
        exit(EXIT_FAILURE);
    }
}


torch::Tensor healpix_rfft_cuda(torch::Tensor f, int L, int nside) {

    int ntheta = 4 * nside - 1;
    int padding = 8 * nside;

    auto device = f.device();
    torch::Dtype ctype;

    // Retrieve the current CUDA stream
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // Use CUDAStreamGuard to set the current stream (though it's already set, this makes it explicit)
    at::cuda::CUDAStreamGuard guard(stream);


    if (f.scalar_type() == torch::kDouble){
        using scalar_t = double;
        ctype = torch::kComplexDouble;

    }else if (f.scalar_type() == torch::kFloat){

        using scalar_t = float;
        ctype = torch::kComplexFloat;
    }
    else{
        AT_ERROR("Unsupported data type for f tensor: ", f.scalar_type());
    }


    auto ftm = torch::zeros({ntheta, L}, torch::dtype(ctype).device(device));
    auto x_pad = torch::zeros({ntheta, padding}, torch::dtype(ctype).device(device));
    auto y_pad = torch::zeros({ntheta, padding}, torch::dtype(ctype).device(device));

    rfft_pre_process_dispatch(x_pad, y_pad, f, padding, nside, stream);

    cufftHandle plan;

    if (f.scalar_type() == torch::kDouble){
        checkCuFFTError(cufftPlan1d(&plan, padding, CUFFT_Z2Z, ntheta));

        checkCuFFTError(cufftSetStream(plan, stream.stream()));

        checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()),
            reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));
        checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(y_pad.data_ptr<c10::complex<double>>()),
            reinterpret_cast<cufftDoubleComplex*>(y_pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));

        x_pad.mul_(y_pad);

        checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()),
            reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), CUFFT_INVERSE));

        x_pad.div_(padding);
        checkCuFFTError(cufftDestroy(plan));

    } else if (f.scalar_type() == torch::kFloat){

        checkCuFFTError(cufftPlan1d(&plan, padding, CUFFT_C2C, ntheta));

        checkCuFFTError(cufftSetStream(plan, stream.stream()));

        checkCuFFTError(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(x_pad.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cufftComplex*>(x_pad.data_ptr<c10::complex<float>>()), CUFFT_FORWARD));
        checkCuFFTError(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(y_pad.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cufftComplex*>(y_pad.data_ptr<c10::complex<float>>()), CUFFT_FORWARD));

        x_pad.mul_(y_pad);

        checkCuFFTError(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(x_pad.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cufftComplex*>(x_pad.data_ptr<c10::complex<float>>()), CUFFT_INVERSE));

        x_pad.div_(padding);
        checkCuFFTError(cufftDestroy(plan));
    }


    rfft_post_process_dispatch(x_pad, ftm, L, padding, nside, stream);

    rfft_phase_shift_dispatch(ftm, L, nside, stream);

    return ftm;

}


torch::Tensor healpix_irfft_cuda(torch::Tensor ftm, int L, int nside) {

    int ntheta = ftm.size(0);
    int padding = 8 * nside;

    auto device = ftm.device();

    torch::Dtype ctype;
    torch::Dtype ftype;

    // Retrieve the current CUDA stream
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // Use CUDAStreamGuard to set the current stream (though it's already set, this makes it explicit)
    at::cuda::CUDAStreamGuard guard(stream);


    if (ftm.scalar_type() == torch::kComplexDouble){

        using scalar_t = double;
        ctype = torch::kComplexDouble;
        ftype = torch::kDouble;

    } else if (ftm.scalar_type() == torch::kComplexFloat){
        using scalar_t = float;
        ctype = torch::kComplexFloat;
        ftype = torch::kFloat;
    } else {

        AT_ERROR("Unsupported data type for ftm tensor: ", ftm.scalar_type());
    }


    auto f = torch::zeros({12 * nside * nside}, torch::dtype(ftype).device(device));
    auto x_pad = torch::zeros({ntheta, padding}, torch::dtype(ctype).device(device));
    auto y_pad = torch::zeros({ntheta, padding}, torch::dtype(ctype).device(device));


    // Launch CUDA kernel for phase shift
    irfft_phase_shift_dispatch(ftm, L, nside, stream);

    // Launch CUDA kernel for processing fm chunks
    irfft_pre_process_dispatch(ftm, x_pad, y_pad, L, padding, nside, stream);

    // Perform FFT on x_pad and y_pad
    cufftHandle plan;


    if (ftm.scalar_type() == torch::kComplexDouble){

        checkCuFFTError(cufftPlan1d(&plan, padding, CUFFT_Z2Z, ntheta));

        checkCuFFTError(cufftSetStream(plan, stream.stream()));

        checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()),
            reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));
        checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(y_pad.data_ptr<c10::complex<double>>()),
            reinterpret_cast<cufftDoubleComplex*>(y_pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));

        x_pad.mul_(y_pad);
        checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()),
            reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), CUFFT_INVERSE));
        x_pad.div_(padding);
        checkCuFFTError(cufftDestroy(plan));

    } else if (ftm.scalar_type() == torch::kComplexFloat){

        checkCuFFTError(cufftPlan1d(&plan, padding, CUFFT_C2C, ntheta));

        checkCuFFTError(cufftSetStream(plan, stream.stream()));

        checkCuFFTError(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(x_pad.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cufftComplex*>(x_pad.data_ptr<c10::complex<float>>()), CUFFT_FORWARD));
        checkCuFFTError(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(y_pad.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cufftComplex*>(y_pad.data_ptr<c10::complex<float>>()), CUFFT_FORWARD));

        x_pad.mul_(y_pad);

        checkCuFFTError(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(x_pad.data_ptr<c10::complex<float>>()),
            reinterpret_cast<cufftComplex*>(x_pad.data_ptr<c10::complex<float>>()), CUFFT_INVERSE));

        x_pad.div_(padding);
        checkCuFFTError(cufftDestroy(plan));
    }


    // Launch CUDA kernel to compute the final result
    irfft_post_process_dispatch(x_pad, f, nside, padding, stream);
    return f;
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("healpix_rfft_cuda", &healpix_rfft_cuda, "HEALPix RFFT with cuFFT and CUDA");
//     m.def("healpix_irfft_cuda", &healpix_irfft_cuda, "HEALPix IRFFT with cuFFT and CUDA");
// }

int nphi_ring(int t, int nside);
int cumulative_nphi_ring(int t, int nside);
double p2phi_ring(int t, int offset, int nside);


torch::Tensor healpix_rfft_cufft(torch::Tensor f, int L, int nside) {
    int ntheta = 4 * nside - 1;
    int padding = 8 * nside;

    auto device = f.device();

    auto ftm = torch::zeros({ntheta, L}, torch::dtype(torch::kComplexDouble).device(device));
    auto x_pad = torch::zeros({ntheta, padding}, torch::dtype(torch::kComplexDouble).device(device));
    auto y_pad = torch::zeros({ntheta, padding}, torch::dtype(torch::kComplexDouble).device(device));

    c10::complex<double> imag_unit(0.0, 1.0);

    for (int t = 0; t < ntheta; ++t) {
        int nphi = nphi_ring(t, nside);
        int index = cumulative_nphi_ring(t, nside);

        auto vec = f.slice(0, index, index + nphi);
        auto coef_arr = M_PI * torch::pow(torch::arange(nphi, torch::dtype(torch::kDouble).device(device)), 2) / nphi * imag_unit;

        auto chirp_b = torch::exp(coef_arr);
        auto chirp_a = 1.0 / chirp_b;

        x_pad.index_put_({t, torch::indexing::Slice(0, nphi)}, vec * chirp_b);
        y_pad.index_put_({t, torch::indexing::Slice(0, nphi)}, chirp_a);
        y_pad.index_put_({t, torch::indexing::Slice(padding - nphi + 1, torch::indexing::None)}, torch::flip(chirp_a.slice(0, 1, torch::indexing::None), {0}));
    }

    // Prepare for cuFFT
    cufftHandle plan;
    checkCuFFTError(cufftPlan1d(&plan, padding, CUFFT_Z2Z, ntheta));

    // Perform FFT on x_pad and y_pad
    checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()),
        reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));
    checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(y_pad.data_ptr<c10::complex<double>>()),
        reinterpret_cast<cufftDoubleComplex*>(y_pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));

    // Element-wise multiplication
    x_pad.mul_(y_pad);

    // Perform IFFT on x_pad
    checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()),
        reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), CUFFT_INVERSE));

    // Normalize IFFT result
    x_pad.div_(padding);

    // Destroy cuFFT plan
    checkCuFFTError(cufftDestroy(plan));

    for (int t = 0; t < ntheta; ++t) {
        int nphi = nphi_ring(t, nside);
        auto coef_arr = M_PI * torch::pow(torch::arange(nphi, torch::dtype(torch::kDouble).device(device)), 2) / nphi * imag_unit;
        auto chirp_b = torch::exp(coef_arr);
        auto result = (x_pad.index({t, torch::indexing::Slice(0, nphi)}) * chirp_b).conj();
        ftm.index_put_({t, torch::indexing::Slice(0, std::min(L, nphi / 2 + 1))}, result.slice(0, 0, std::min(L, nphi / 2 + 1)));
    }

    for (int t = 0; t < ntheta; ++t) {
        double phi_ring_offset = p2phi_ring(t, 0, nside);
        auto phase_shift = torch::exp(-imag_unit * phi_ring_offset * torch::arange(L, torch::dtype(torch::kDouble).device(device)));
        ftm.index_put_({t, torch::indexing::Slice()}, ftm.index({t, torch::indexing::Slice()}) * phase_shift);
    }

    return ftm;
}

torch::Tensor healpix_irfft_cufft(torch::Tensor ftm, int L, int nside) {
    auto device = ftm.device();
    auto f = torch::zeros({12 * nside * nside}, torch::dtype(torch::kDouble).device(device));

    int ntheta = ftm.size(0);
    int padding = 8 * nside;

    auto x_pad = torch::zeros({ntheta, padding}, torch::dtype(torch::kComplexDouble).device(device));
    auto y_pad = torch::zeros({ntheta, padding}, torch::dtype(torch::kComplexDouble).device(device));

    c10::complex<double> imag_unit(0.0, 1.0);

    for (int t = 0; t < ntheta; ++t) {
        double phi_ring_offset = p2phi_ring(t, 0, nside);
        auto phase_shift = torch::exp(imag_unit * phi_ring_offset * torch::arange(L, torch::dtype(torch::kDouble).device(device)).to(torch::kComplexDouble));
        ftm.index_put_({t, torch::indexing::Slice()}, ftm.index({t, torch::indexing::Slice()}) * phase_shift);
    }

    for (int t = 0; t < ntheta; ++t) {
        int nphi = nphi_ring(t, nside);
        int index = cumulative_nphi_ring(t, nside);

        auto fm_chunk = torch::zeros({nphi / 2 + 1}, torch::dtype(torch::kComplexDouble).device(device));
        fm_chunk.index_put_({torch::indexing::Slice(0, std::min(nphi / 2 + 1, L))}, ftm.index({t, torch::indexing::Slice(0, std::min(nphi / 2 + 1, L))}));
        fm_chunk = torch::cat({fm_chunk, fm_chunk.slice(0, 1, -1).conj().flip(0)}).conj();

        auto coef_arr = M_PI * torch::pow(torch::arange(nphi, torch::dtype(torch::kDouble).device(device)), 2) / nphi * imag_unit;

        auto chirp_a = torch::exp(coef_arr);
        auto chirp_b = torch::exp(-coef_arr);

        x_pad.index_put_({t, torch::indexing::Slice(0, nphi)}, fm_chunk * chirp_b);
        y_pad.index_put_({t, torch::indexing::Slice(0, nphi)}, chirp_a);
        y_pad.index_put_({t, torch::indexing::Slice(padding - nphi + 1, torch::indexing::None)}, torch::flip(chirp_a.slice(0, 1, torch::indexing::None), {0}));
    }

    // Prepare for cuFFT
    cufftHandle plan;
    checkCuFFTError(cufftPlan1d(&plan, padding, CUFFT_Z2Z, ntheta));

    // Perform FFT on x_pad and y_pad
    checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));
    checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(y_pad.data_ptr<c10::complex<double>>()), reinterpret_cast<cufftDoubleComplex*>(y_pad.data_ptr<c10::complex<double>>()), CUFFT_FORWARD));

    // Element-wise multiplication
    x_pad.mul_(y_pad);

    // Perform IFFT on x_pad
    checkCuFFTError(cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), reinterpret_cast<cufftDoubleComplex*>(x_pad.data_ptr<c10::complex<double>>()), CUFFT_INVERSE));

    // Normalize IFFT result
    x_pad.div_(padding);

    // Destroy cuFFT plan
    checkCuFFTError(cufftDestroy(plan));

    for (int t = 0; t < ntheta; ++t) {
        int nphi = nphi_ring(t, nside);
        int index = cumulative_nphi_ring(t, nside);
        auto coef_arr = M_PI * torch::pow(torch::arange(nphi, torch::dtype(torch::kDouble).device(device)), 2) / nphi * imag_unit;
        auto chirp_b = torch::exp(-coef_arr);
        auto result = torch::real(x_pad.index({t, torch::indexing::Slice(0, nphi)}) * chirp_b);
        f.index_put_({torch::indexing::Slice(index, index + nphi)}, result);
    }

    return f;
}


torch::Tensor healpix_rfft(torch::Tensor f, int L, int nside) {
    int ntheta = 4 * nside - 1;
    int padding = 8 * nside;

    auto device = f.device();

    auto ftm = torch::zeros({ntheta, L}, torch::dtype(torch::kComplexDouble).device(device));
    auto x_pad = torch::zeros({ntheta, padding}, torch::dtype(torch::kComplexDouble).device(device));
    auto y_pad = torch::zeros({ntheta, padding}, torch::dtype(torch::kComplexDouble).device(device));

    c10::complex<double> imag_unit(0.0, 1.0);

    for (int t = 0; t < ntheta; ++t) {
        int nphi = nphi_ring(t, nside);
        int index = cumulative_nphi_ring(t, nside);

        auto vec = f.slice(0, index, index + nphi);
        auto coef_arr = M_PI * torch::pow(torch::arange(nphi, torch::dtype(torch::kDouble).device(device)), 2) / nphi * imag_unit;

        auto chirp_b = torch::exp(coef_arr);
        auto chirp_a = 1.0 / chirp_b;

        x_pad.index_put_({t, torch::indexing::Slice(0, nphi)}, vec * chirp_b);
        y_pad.index_put_({t, torch::indexing::Slice(0, nphi)}, chirp_a);
        y_pad.index_put_({t, torch::indexing::Slice(padding - nphi + 1, torch::indexing::None)}, torch::flip(chirp_a.slice(0, 1, torch::indexing::None), {0}));
    }

    // Conv
    x_pad = torch::fft::fft(x_pad, x_pad.size(-1));
    y_pad = torch::fft::fft(y_pad, y_pad.size(-1));
    x_pad *= y_pad;
    x_pad = torch::fft::ifft(x_pad, x_pad.size(-1));

    for (int t = 0; t < ntheta; ++t) {
        int nphi = nphi_ring(t, nside);
        auto coef_arr = M_PI * torch::pow(torch::arange(nphi, torch::dtype(torch::kDouble).device(device)), 2) / nphi * imag_unit;
        auto chirp_b = torch::exp(coef_arr);
        auto result = (x_pad.index({t, torch::indexing::Slice(0, nphi)}) * chirp_b).conj();
        ftm.index_put_({t, torch::indexing::Slice(0, std::min(L, nphi / 2 + 1))}, result.slice(0, 0, std::min(L, nphi / 2 + 1)));
    }

    for (int t = 0; t < ntheta; ++t) {
        double phi_ring_offset = p2phi_ring(t, 0, nside);
        auto phase_shift = torch::exp(-imag_unit * phi_ring_offset * torch::arange(L, torch::dtype(torch::kDouble).device(device)));
        ftm.index_put_({t, torch::indexing::Slice()}, ftm.index({t, torch::indexing::Slice()}) * phase_shift);
    }

    return ftm;
}

torch::Tensor healpix_irfft(torch::Tensor ftm, int L, int nside) {
    auto device = ftm.device();
    auto f = torch::zeros({12 * nside * nside}, torch::dtype(torch::kDouble).device(device));

    int ntheta = ftm.size(0);
    int padding = 8 * nside;

    auto x_pad = torch::zeros({ntheta, padding}, torch::dtype(torch::kComplexDouble).device(device));
    auto y_pad = torch::zeros({ntheta, padding}, torch::dtype(torch::kComplexDouble).device(device));

    c10::complex<double> imag_unit(0.0, 1.0);

    for (int t = 0; t < ntheta; ++t) {
        double phi_ring_offset = p2phi_ring(t, 0, nside);
        auto phase_shift = torch::exp(imag_unit * phi_ring_offset * torch::arange(L, torch::dtype(torch::kDouble).device(device)).to(torch::kComplexDouble));
        ftm.index_put_({t, torch::indexing::Slice()}, ftm.index({t, torch::indexing::Slice()}) * phase_shift);
    }

    for (int t = 0; t < ntheta; ++t) {
        int nphi = nphi_ring(t, nside);
        int index = cumulative_nphi_ring(t, nside);

        auto fm_chunk = torch::zeros({nphi / 2 + 1}, torch::dtype(torch::kComplexDouble).device(device));
        fm_chunk.index_put_({torch::indexing::Slice(0, std::min(nphi / 2 + 1, L))}, ftm.index({t, torch::indexing::Slice(0, std::min(nphi / 2 + 1, L))}));
        fm_chunk = torch::cat({fm_chunk, fm_chunk.slice(0, 1, -1).conj().flip(0)}).conj();

        auto coef_arr = M_PI * torch::pow(torch::arange(nphi, torch::dtype(torch::kDouble).device(device)), 2) / nphi * imag_unit;

        auto chirp_a = torch::exp(coef_arr);
        auto chirp_b = torch::exp(-coef_arr);

        x_pad.index_put_({t, torch::indexing::Slice(0, nphi)}, fm_chunk * chirp_b);
        y_pad.index_put_({t, torch::indexing::Slice(0, nphi)}, chirp_a);
        y_pad.index_put_({t, torch::indexing::Slice(padding - nphi + 1, torch::indexing::None)}, torch::flip(chirp_a.slice(0, 1, torch::indexing::None), {0}));
    }

    // Conv
    x_pad = torch::fft::fft(x_pad, x_pad.size(-1));
    y_pad = torch::fft::fft(y_pad, y_pad.size(-1));
    x_pad *= y_pad;
    x_pad = torch::fft::ifft(x_pad, x_pad.size(-1));

    for (int t = 0; t < ntheta; ++t) {
        int nphi = nphi_ring(t, nside);
        int index = cumulative_nphi_ring(t, nside);
        auto coef_arr = M_PI * torch::pow(torch::arange(nphi, torch::dtype(torch::kDouble).device(device)), 2) / nphi * imag_unit;
        auto chirp_b = torch::exp(-coef_arr);
        auto result = torch::real(x_pad.index({t, torch::indexing::Slice(0, nphi)}) * chirp_b);
        f.index_put_({torch::indexing::Slice(index, index + nphi)}, result);
    }

    return f;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("healpix_rfft", &healpix_rfft, "HEALPix RFFT");
    m.def("healpix_irfft", &healpix_irfft, "HEALPix IRFFT");
    m.def("healpix_rfft_cufft", &healpix_rfft_cufft, "HEALPix RFFT with cuFFT");
    m.def("healpix_irfft_cufft", &healpix_irfft_cufft, "HEALPix IRFFT with cuFFT");
    m.def("healpix_rfft_cuda", &healpix_rfft_cuda, "HEALPix RFFT with cuFFT and CUDA");
    m.def("healpix_irfft_cuda", &healpix_irfft_cuda, "HEALPix IRFFT with cuFFT and CUDA");
    m.def("healpix_rfft_class", &healpix_rfft_class, "HEALPix RFFT in class");
    m.def("healpix_irfft_class", &healpix_irfft_class, "HEALPix IRFFT in class");
    m.def("healpix_rfft_batch", &healpix_rfft_batch, "HEALPix RFFT in batch");
    m.def("healpix_irfft_batch", &healpix_irfft_batch, "HEALPix IRFFT in batch");
}


// CUDA kernel for computing the number of phi samples for each theta ring
int nphi_ring(int t, int nside) {
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


int cumulative_nphi_ring(int t, int nside) {
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

double p2phi_ring(int t, int p, int nside) {
    // Convert index to phi angle for HEALPix
    // t: theta, index of ring
    // p: phi, index within ring

    double shift = 0.5;
    double factor;

    if ((t + 1 >= nside) && (t + 1 <= 3 * nside)) {
        shift *= (t - nside + 2) % 2;
        factor = M_PI / (2 * nside);
    } else if (t + 1 > 3 * nside) {
        factor = M_PI / (2 * (4 * nside - t - 1));
    } else {
        factor = M_PI / (2 * (t + 1));
    }

    return factor * (p + shift);
}


template<typename I> inline int compute_order(I nside) {

    unsigned int res = 0;
    while (nside > 0x00FF) {res |= 8; nside >>= 8;}
    if (nside > 0x000F) {res |= 4; nside >>= 4;}
    if (nside > 0x0003) {res |= 2;nside >>= 2;}
    if (nside > 0x0001) {res |= 1;}
    return res;
}
