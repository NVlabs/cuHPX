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

#ifndef HPX_REMAPPING_H
#define HPX_REMAPPING_H

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/extension.h>

// Dispatch functions for each wrapper
void ring2nest_dispatch(torch::Tensor data_in_ring, torch::Tensor data_in_nest, const int nside, const size_t num_elements);

void nest2ring_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_ring, const int nside, const size_t num_elements);

void nest2xy_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void xy2nest_dispatch(torch::Tensor data_in_xy, torch::Tensor data_in_nest, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void ring2xy_dispatch(torch::Tensor data_in_ring, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void xy2ring_dispatch(torch::Tensor data_in_xy, torch::Tensor data_in_ring, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void xy2xy_dispatch(torch::Tensor data_xy_in, torch::Tensor data_xy_out, const int src_origin, const bool src_clockwise,
                    const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void benchmark_nest_ring_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_ring, const int nside, const size_t num_elements);


void ring2nest_batch_dispatch(torch::Tensor data_in_ring, torch::Tensor data_in_nest, const int nside, const size_t num_elements);

void nest2ring_batch_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_ring, const int nside, const size_t num_elements);

void nest2xy_batch_dispatch(torch::Tensor data_in_nest, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void xy2nest_batch_dispatch(torch::Tensor data_in_xy, torch::Tensor data_in_nest, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void ring2xy_batch_dispatch(torch::Tensor data_in_ring, torch::Tensor data_in_xy, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void xy2ring_batch_dispatch(torch::Tensor data_in_xy, torch::Tensor data_in_ring, const int src_origin, const bool src_clockwise,
                      const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

void xy2xy_batch_dispatch(torch::Tensor data_xy_in, torch::Tensor data_xy_out, const int src_origin, const bool src_clockwise,
                    const int dest_origin, const bool dest_clockwise, const int nside, const size_t num_elements);

#endif // HPX_REMAPPING_H
