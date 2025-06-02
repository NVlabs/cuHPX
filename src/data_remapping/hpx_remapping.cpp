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

#include <torch/extension.h> 
#include <torch/torch.h>
#include "hpx_remapping.h"

torch::Tensor ring2nest(torch::Tensor data_in_ring, const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;
    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_nest = torch::empty_like(data_in_ring);

    ring2nest_dispatch(data_in_ring, data_in_nest, nside, num_elements);

    return data_in_nest;
}


torch::Tensor nest2ring(torch::Tensor data_in_nest, const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_ring = torch::empty_like(data_in_nest);

    nest2ring_dispatch(data_in_nest, data_in_ring, nside, num_elements);

    return data_in_ring;
}

torch::Tensor nest2xy(torch::Tensor data_in_nest, const std::string& origin, const bool clockwise, 
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_xy = torch::empty_like(data_in_nest);

    int src_origin = 0;
    int dest_origin;

    bool src_clockwise = false;
    bool dest_clockwise = clockwise;

    if (origin == "S"){
        dest_origin = 0;
    } else if (origin == "E"){
        dest_origin = 1;
    } else if (origin == "N"){
        dest_origin = 2;
    } else if (origin == "W"){
        dest_origin = 3;
    } else {
        TORCH_CHECK(false, "The origin must be one from S, E, N, W. Stop");
    }


    nest2xy_dispatch(data_in_nest, data_in_xy, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_in_xy;
}

torch::Tensor xy2nest(torch::Tensor data_in_xy, const std::string& origin, const bool clockwise, 
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_nest = torch::empty_like(data_in_xy);


    int src_origin;
    int dest_origin = 0;

    bool src_clockwise = clockwise;
    bool dest_clockwise = false;

    if (origin == "S"){
        src_origin = 0;
    } else if (origin == "E"){
        src_origin = 1;
    } else if (origin == "N"){
        src_origin = 2;
    } else if (origin == "W"){
        src_origin = 3;
    } else {
        TORCH_CHECK(false, "The origin must be one from S, E, N, W. Stop");
    }


    xy2nest_dispatch(data_in_xy, data_in_nest, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_in_nest;
}

torch::Tensor ring2xy(torch::Tensor data_in_ring, const std::string& origin, const bool clockwise, 
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_xy = torch::empty_like(data_in_ring);

    int src_origin = 0;
    int dest_origin;

    bool src_clockwise = false;
    bool dest_clockwise = clockwise;

    if (origin == "S"){
        dest_origin = 0;
    } else if (origin == "E"){
        dest_origin = 1;
    } else if (origin == "N"){
        dest_origin = 2;
    } else if (origin == "W"){
        dest_origin = 3;
    } else {
        TORCH_CHECK(false, "The origin must be one from S, E, N, W. Stop");
    }


    ring2xy_dispatch(data_in_ring, data_in_xy, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_in_xy;
}

torch::Tensor xy2ring(torch::Tensor data_in_xy, const std::string& origin, const bool clockwise, 
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_ring = torch::empty_like(data_in_xy);

    int src_origin;
    int dest_origin = 0;

    bool src_clockwise = clockwise;
    bool dest_clockwise = false;

    if (origin == "S"){
        src_origin = 0;
    } else if (origin == "E"){
        src_origin = 1;
    } else if (origin == "N"){
        src_origin = 2;
    } else if (origin == "W"){
        src_origin = 3;
    } else {
        TORCH_CHECK(false, "The origin must be one from S, E, N, W. Stop");
    }


    xy2ring_dispatch(data_in_xy, data_in_ring, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_in_ring;
}

torch::Tensor xy2xy(torch::Tensor data_xy_in, const std::string& s_origin, const bool src_clockwise, const std::string& d_origin, const bool dest_clockwise,
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_xy_out = torch::empty_like(data_xy_in);

    int src_origin;
    int dest_origin;


    if (s_origin == "S"){
        src_origin = 0;
    } else if (s_origin == "E"){
        src_origin = 1;
    } else if (s_origin == "N"){
        src_origin = 2;
    } else if (s_origin == "W"){
        src_origin = 3;
    } else {
        TORCH_CHECK(false, "The src origin must be one from S, E, N, W. Stop");
    }

    if (d_origin == "S"){
        dest_origin = 0;
    } else if (d_origin == "E"){
        dest_origin = 1;
    } else if (d_origin == "N"){
        dest_origin = 2;
    } else if (d_origin == "W"){
        dest_origin = 3;
    } else {
        TORCH_CHECK(false, "The dest origin must be one from S, E, N, W. Stop");
    }


    xy2xy_dispatch(data_xy_in, data_xy_out, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_xy_out;
}


torch::Tensor ring2nest_batch(torch::Tensor data_in_ring, const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;
    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_nest = torch::empty_like(data_in_ring);

    ring2nest_batch_dispatch(data_in_ring, data_in_nest, nside, num_elements);

    return data_in_nest;
}


torch::Tensor nest2ring_batch(torch::Tensor data_in_nest, const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_ring = torch::empty_like(data_in_nest);

    nest2ring_batch_dispatch(data_in_nest, data_in_ring, nside, num_elements);

    return data_in_ring;
}

torch::Tensor nest2xy_batch(torch::Tensor data_in_nest, const std::string& origin, const bool clockwise, 
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_xy = torch::empty_like(data_in_nest);

    int src_origin = 0;
    int dest_origin;

    bool src_clockwise = false;
    bool dest_clockwise = clockwise;

    if (origin == "S"){
        dest_origin = 0;
    } else if (origin == "E"){
        dest_origin = 1;
    } else if (origin == "N"){
        dest_origin = 2;
    } else if (origin == "W"){
        dest_origin = 3;
    } else {
        TORCH_CHECK(false, "The origin must be one from S, E, N, W. Stop");
    }


    nest2xy_batch_dispatch(data_in_nest, data_in_xy, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_in_xy;
}

torch::Tensor xy2nest_batch(torch::Tensor data_in_xy, const std::string& origin, const bool clockwise, 
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_nest = torch::empty_like(data_in_xy);


    int src_origin;
    int dest_origin = 0;

    bool src_clockwise = clockwise;
    bool dest_clockwise = false;

    if (origin == "S"){
        src_origin = 0;
    } else if (origin == "E"){
        src_origin = 1;
    } else if (origin == "N"){
        src_origin = 2;
    } else if (origin == "W"){
        src_origin = 3;
    } else {
        TORCH_CHECK(false, "The origin must be one from S, E, N, W. Stop");
    }


    xy2nest_batch_dispatch(data_in_xy, data_in_nest, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_in_nest;
}

torch::Tensor ring2xy_batch(torch::Tensor data_in_ring, const std::string& origin, const bool clockwise, 
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_xy = torch::empty_like(data_in_ring);

    int src_origin = 0;
    int dest_origin;

    bool src_clockwise = false;
    bool dest_clockwise = clockwise;

    if (origin == "S"){
        dest_origin = 0;
    } else if (origin == "E"){
        dest_origin = 1;
    } else if (origin == "N"){
        dest_origin = 2;
    } else if (origin == "W"){
        dest_origin = 3;
    } else {
        TORCH_CHECK(false, "The origin must be one from S, E, N, W. Stop");
    }


    ring2xy_batch_dispatch(data_in_ring, data_in_xy, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_in_xy;
}

torch::Tensor xy2ring_batch(torch::Tensor data_in_xy, const std::string& origin, const bool clockwise, 
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_in_ring = torch::empty_like(data_in_xy);

    int src_origin;
    int dest_origin = 0;

    bool src_clockwise = clockwise;
    bool dest_clockwise = false;

    if (origin == "S"){
        src_origin = 0;
    } else if (origin == "E"){
        src_origin = 1;
    } else if (origin == "N"){
        src_origin = 2;
    } else if (origin == "W"){
        src_origin = 3;
    } else {
        TORCH_CHECK(false, "The origin must be one from S, E, N, W. Stop");
    }


    xy2ring_batch_dispatch(data_in_xy, data_in_ring, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_in_ring;
}

torch::Tensor xy2xy_batch(torch::Tensor data_xy_in, const std::string& s_origin, const bool src_clockwise, const std::string& d_origin, const bool dest_clockwise,
        const int nside, const size_t num_elements) {

    const size_t expected_num_elements = static_cast<size_t>(nside)*nside*12;

    // Check if num_elements matches the expected number of elements
    TORCH_CHECK(num_elements == expected_num_elements, "The number of elements in the input array is not equal to the number of HEALPix grid at current nside. Stop.");

    auto data_xy_out = torch::empty_like(data_xy_in);

    int src_origin;
    int dest_origin;


    if (s_origin == "S"){
        src_origin = 0;
    } else if (s_origin == "E"){
        src_origin = 1;
    } else if (s_origin == "N"){
        src_origin = 2;
    } else if (s_origin == "W"){
        src_origin = 3;
    } else {
        TORCH_CHECK(false, "The src origin must be one from S, E, N, W. Stop");
    }

    if (d_origin == "S"){
        dest_origin = 0;
    } else if (d_origin == "E"){
        dest_origin = 1;
    } else if (d_origin == "N"){
        dest_origin = 2;
    } else if (d_origin == "W"){
        dest_origin = 3;
    } else {
        TORCH_CHECK(false, "The dest origin must be one from S, E, N, W. Stop");
    }


    xy2xy_batch_dispatch(data_xy_in, data_xy_out, src_origin, src_clockwise, dest_origin, dest_clockwise, nside, num_elements);

    return data_xy_out;
}



PYBIND11_MODULE(cuhpx_remap, m) {

    m.def("ring2nest", &ring2nest, "Convert ring to nest (CUDA)");
    m.def("nest2ring", &nest2ring, "Convert nest to ring (CUDA)");

    m.def("nest2xy", &nest2xy, "Convert nest to xy (CUDA)");
    m.def("xy2nest", &xy2nest, "Convert xy to nest (CUDA)");

    m.def("ring2xy", &ring2xy, "Convert ring to xy (CUDA)");
    m.def("xy2ring", &xy2ring, "Convert xy to ring (CUDA)");

    m.def("xy2xy", &xy2xy, "Convert xy to xy (CUDA)");

    m.def("ring2nest_batch", &ring2nest_batch, "Convert ring to nest (CUDA) in batch");
    m.def("nest2ring_batch", &nest2ring_batch, "Convert nest to ring (CUDA) in batch");

    m.def("nest2xy_batch", &nest2xy_batch, "Convert nest to xy (CUDA) in batch");
    m.def("xy2nest_batch", &xy2nest_batch, "Convert xy to nest (CUDA) in batch");

    m.def("ring2xy_batch", &ring2xy_batch, "Convert ring to xy (CUDA) in batch");
    m.def("xy2ring_batch", &xy2ring_batch, "Convert xy to ring (CUDA) in batch");

    m.def("xy2xy_batch", &xy2xy_batch, "Convert xy to xy (CUDA) in batch");

}

