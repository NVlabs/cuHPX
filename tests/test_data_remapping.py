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

import healpy as hp
import torch

import cuhpx as hpx

# Read the order value from user input
nside = int(input("Enter the nside value: "))
nelements = 12 * nside**2

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Function to test ring2nest and nest2ring for a given dtype
def test_ring2nest_nest2ring(dtype):
    # Generate the input tensor in RING ordering
    tensor_in_ring = torch.arange(nelements, device=device, dtype=dtype)

    # Print the first five elements of the input tensor for ring2nest
    print(f"Input tensor (RING, dtype={dtype}) first 5 elements:", tensor_in_ring[:5])

    # Use hpx.ring2nest to convert to NESTED ordering
    tensor_in_nest = hpx.ring2nest(tensor_in_ring, nside)
    result_tensor_hpx_nest = tensor_in_nest.to('cpu')

    # Use healpy.pixelfunc.reorder to convert to NESTED ordering
    map_in_ring = tensor_in_ring.cpu().numpy()
    result_tensor_healpy_nest = torch.tensor(hp.pixelfunc.reorder(map_in_ring, inp='RING', out='NESTED'), dtype=dtype)

    # Compare the results
    comparison_ring2nest = torch.equal(result_tensor_hpx_nest, result_tensor_healpy_nest)
    print(f"Are the ring2nest results identical (dtype={dtype})?", comparison_ring2nest)
    print(f"HPX ring2nest result (dtype={dtype}) first 5 elements:", result_tensor_hpx_nest[:5])
    print(f"Healpy ring2nest result (dtype={dtype}) first 5 elements:", result_tensor_healpy_nest[:5])

    # Generate the input tensor in NEST ordering for nest2ring
    tensor_in_nest = torch.arange(nelements, device=device, dtype=dtype)

    # Print the first five elements of the input tensor for nest2ring
    print(f"Input tensor (NEST, dtype={dtype}) first 5 elements:", tensor_in_nest[:5])

    # Use hpx.nest2ring to convert to RING ordering
    tensor_in_ring = hpx.nest2ring(tensor_in_nest, nside)
    result_tensor_hpx_ring = tensor_in_ring.to('cpu')

    # Use healpy.pixelfunc.reorder to convert to RING ordering
    map_in_nest = tensor_in_nest.cpu().numpy()
    result_tensor_healpy_ring = torch.tensor(hp.pixelfunc.reorder(map_in_nest, inp='NESTED', out='RING'), dtype=dtype)

    # Compare the results
    comparison_nest2ring = torch.equal(result_tensor_hpx_ring, result_tensor_healpy_ring)
    print(f"Are the nest2ring results identical (dtype={dtype})?", comparison_nest2ring)
    print(f"HPX nest2ring result (dtype={dtype}) first 5 elements:", result_tensor_hpx_ring[:5])
    print(f"Healpy nest2ring result (dtype={dtype}) first 5 elements:", result_tensor_healpy_ring[:5])


# Test for int32
test_ring2nest_nest2ring(torch.int32)

# Test for float64 (double)
test_ring2nest_nest2ring(torch.float64)
