cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuhpx LANGUAGES CXX CUDA)

# C++ and CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python and determine site-packages path
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Add PyTorch CMake path
list(APPEND CMAKE_PREFIX_PATH "${PYTHON_SITE_PACKAGES}/torch/share/cmake")

# Find required packages
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

# Optional: Locate libtorch_python.so dynamically
set(TORCH_PYTHON_LIB "${PYTHON_SITE_PACKAGES}/torch/lib/libtorch_python.so")

# Set CUDA architecture (change if needed)
set(CMAKE_CUDA_ARCHITECTURES 90)
list(APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr --expt-extended-lambda -O3)

# Include paths
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ==================== cuhpx_fft ====================
set(CUHPX_FFT_SRC
    src/harmonic_transform/hpx_fft.cpp
    src/harmonic_transform/hpx_fft_cuda.cu
)

pybind11_add_module(cuhpx_fft MODULE ${CUHPX_FFT_SRC})
target_link_libraries(cuhpx_fft PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(cuhpx_fft PRIVATE ${TORCH_PYTHON_LIB})
target_compile_definitions(cuhpx_fft PRIVATE ${TORCH_CXX_FLAGS})
set_target_properties(cuhpx_fft PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cuhpx)

# ==================== cuhpx_remap ====================
set(CUHPX_REMAP_SRC
    src/data_remapping/hpx_remapping.cpp
    src/data_remapping/hpx_remapping_cuda.cu
)

pybind11_add_module(cuhpx_remap MODULE ${CUHPX_REMAP_SRC})
target_link_libraries(cuhpx_remap PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(cuhpx_remap PRIVATE ${TORCH_PYTHON_LIB})
target_compile_definitions(cuhpx_remap PRIVATE ${TORCH_CXX_FLAGS})
set_target_properties(cuhpx_remap PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cuhpx)

# ==================== Install ====================
install(TARGETS cuhpx_fft cuhpx_remap DESTINATION cuhpx)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cuhpx/data DESTINATION cuhpx FILES_MATCHING PATTERN "*.fits")

message(STATUS "✅ cuhpx build system configured successfully.")


