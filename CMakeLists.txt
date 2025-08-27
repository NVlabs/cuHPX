cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuhpx LANGUAGES CXX CUDA)

# ---------- Basics ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# LTO/IPO can drop CUDA fatbins; keep it off for these extensions
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# CUDA archs default; override via -DCMAKE_CUDA_ARCHITECTURES=80;86;90
if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
  set(CMAKE_CUDA_ARCHITECTURES 80;86;90)  # Ampere and Hopper default
endif()
message(STATUS "✅ cuhpx build configured. CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

# ---------- Python / Torch ----------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Find site-packages so we can locate Torch's CMake configs
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYTHON_SITE_PACKAGES}/torch/share/cmake")

find_package(Torch REQUIRED)        # ${TORCH_LIBRARIES}, ${TORCH_CXX_FLAGS}
find_package(pybind11 REQUIRED)

# ---------- CUDA toolchain (modern imported targets) ----------
find_package(CUDAToolkit REQUIRED)  # provides CUDA::cudart, etc.

# Optional: Torch's Python shim (resolves at::Tensor pybind casters)
find_library(TORCH_PYTHON_LIBRARY
  NAMES torch_python
  HINTS "${PYTHON_SITE_PACKAGES}/torch/lib" "${TORCH_INSTALL_PREFIX}/lib"
)
if(TORCH_PYTHON_LIBRARY)
  message(STATUS "Found torch_python at: ${TORCH_PYTHON_LIBRARY}")
endif()

include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ---------- Helper: apply safe CUDA flags per target ----------
function(cuhpx_apply_cuda_flags target_name)
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=20014>
  )
  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION FALSE
  )
endfunction()

# ==================== cuhpx_fft ====================
set(CUHPX_FFT_SRC
  src/harmonic_transform/hpx_fft.cpp
  src/harmonic_transform/hpx_fft_cuda.cu
)
pybind11_add_module(cuhpx_fft MODULE ${CUHPX_FFT_SRC})
target_compile_definitions(cuhpx_fft PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(cuhpx_fft PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  Python::Module
)
if(TORCH_PYTHON_LIBRARY)
  target_link_libraries(cuhpx_fft PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()
set_target_properties(cuhpx_fft PROPERTIES PREFIX "" OUTPUT_NAME "cuhpx_fft")
cuhpx_apply_cuda_flags(cuhpx_fft)

# ==================== cuhpx_remap ====================
set(CUHPX_REMAP_SRC
  src/data_remapping/hpx_remapping.cpp
  src/data_remapping/hpx_remapping_cuda.cu
)
pybind11_add_module(cuhpx_remap MODULE ${CUHPX_REMAP_SRC})
target_compile_definitions(cuhpx_remap PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(cuhpx_remap PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  Python::Module
)
if(TORCH_PYTHON_LIBRARY)
  target_link_libraries(cuhpx_remap PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()
set_target_properties(cuhpx_remap PROPERTIES PREFIX "" OUTPUT_NAME "cuhpx_remap")
cuhpx_apply_cuda_flags(cuhpx_remap)

# ---------- Install layout for wheels ----------
# scikit-build-core will install these into site-packages/cuhpx/ for wheels.
install(TARGETS cuhpx_fft cuhpx_remap LIBRARY DESTINATION cuhpx)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cuhpx/data
  DESTINATION cuhpx
  FILES_MATCHING PATTERN "*.fits"
)
