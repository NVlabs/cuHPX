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



from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuhpx',
    version='0.1.0',
    packages=find_packages(),
    package_data={'cuhpx': ['**/*.fits']},
    url='https://gitlab-master.nvidia.com/Devtech-Compute/cuhpx',
    license='TBD',
    author='Xiaopo Cheng, Akshay Subramaniam',
    author_email='xiaopoc@nvidia.com, asubramaniam@nvidia.com',
    description='A library for performing transformations and analysis on HEALPix',
    install_requires=[
        'numpy',
        'torch',
        'astropy',
        'torch_harmonics',
    ],
    ext_modules=[
        CUDAExtension('cuhpx_remap', sources= [
            'src/data_remapping/hpx_remapping.cpp',
            'src/data_remapping/hpx_remapping_cuda.cu',],
            extra_compile_args={'nvcc': ['-O2']},
            extra_link_args=['-lnvToolsExt']
        ),
        CUDAExtension('cuhpx_fft', sources= [
            'src/harmonic_transform/hpx_fft.cpp',
            'src/harmonic_transform/hpx_fft_cuda.cu',],
            extra_compile_args={'nvcc': ['-O2','-lineinfo']},
            extra_link_args=['-lnvToolsExt']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

