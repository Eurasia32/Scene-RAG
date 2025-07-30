cmake_minimum_required(VERSION 3.21)
project(opensplat_render)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# GPU runtime options
set(GPU_RUNTIME "CUDA" CACHE STRING "HIP or CUDA")
set(OPENSPLAT_USE_FAST_MATH OFF CACHE BOOL "Enable fast math optimizations for GPU kernels (-use_fast_math / -ffast-math)")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Read version
file(READ "VERSION" APP_VERSION)

# Read git commit
set(GIT_REV "")
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" 
                OUTPUT_VARIABLE GIT_REV 
                ERROR_QUIET)
string(REGEX REPLACE "\n$" "" GIT_REV "${GIT_REV}")
if (NOT "${GIT_REV}" STREQUAL "")
    set(DAPP_VERSION "${APP_VERSION} (git commit ${GIT_REV})")
    set(DAPP_REVISION "${GIT_REV}")
else()
    set(DAPP_VERSION "${APP_VERSION}")
    set(DAPP_REVISION "dev")
endif()

message("OpenSplat Render Version: ${DAPP_VERSION}")
add_compile_options("-DAPP_VERSION=\"${DAPP_VERSION}\"")
add_compile_options("-DAPP_REVISION=\"${DAPP_REVISION}\"")

# Policy settings
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# CUDA settings
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -diag-suppress=20012)
if(OPENSPLAT_USE_FAST_MATH)
    message(STATUS "Fast math optimizations enabled for CUDA")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)
endif()

# External dependencies
include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_Declare(nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip
)
FetchContent_MakeAvailable(nlohmann_json nanoflann)

# GPU runtime detection
if(GPU_RUNTIME STREQUAL "CUDA")
    find_package(CUDAToolkit)
    if (NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA toolkit not found, building with CPU support only")
        set(GPU_RUNTIME "CPU")
    else()
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            SET(CMAKE_CUDA_ARCHITECTURES 70;75;80)
        endif()
        set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
        list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "([0-9])([0-9])" "\\1.\\2")
        string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
        message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
    endif()
elseif(GPU_RUNTIME STREQUAL "HIP")
    set(USE_HIP ON CACHE BOOL "Use HIP for GPU acceleration")
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP REQUIRED)
    if(OPENSPLAT_USE_FAST_MATH)
        message(STATUS "Fast math optimizations enabled for HIP")
        set(CMAKE_HIP_FLAGS ${CMAKE_HIP_FLAGS} -ffast-math)
    endif()
    file(GLOB_RECURSE GSPLAT_GPU_SRC LIST_DIRECTORIES False rasterizer/gsplat/*.cu)
    set_source_files_properties(${GSPLAT_GPU_SRC} PROPERTIES LANGUAGE HIP)
    if(WIN32)
        set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
    else()
        set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
else()
    set(GPU_RUNTIME "CPU")
endif()

# Language standards
set(CMAKE_CXX_STANDARD 17)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    enable_language(${GPU_RUNTIME})
    set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
    set(${GPU_RUNTIME}_STANDARD 17)
endif()

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

# Find required libraries
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if (NOT WIN32 AND NOT APPLE)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)

# GPU rasterization libraries
set(GSPLAT_LIBS gsplat_cpu)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    add_library(gsplat rasterizer/gsplat/forward.cu rasterizer/gsplat/backward.cu rasterizer/gsplat/bindings.cu rasterizer/gsplat/helpers.cuh)
    list(APPEND GSPLAT_LIBS gsplat)
    if(GPU_RUNTIME STREQUAL "CUDA")
        set(GPU_LIBRARIES "cuda")
        target_link_libraries(gsplat PUBLIC cuda)
    else(GPU_RUNTIME STREQUAL "HIP")
        set(GPU_INCLUDE_DIRS "${ROCM_ROOT}/include")
        target_compile_definitions(gsplat PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    endif()
    target_include_directories(gsplat PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${TORCH_INCLUDE_DIRS}
    )
    
    if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
        FetchContent_Declare(glm
            URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
        )
        FetchContent_MakeAvailable(glm)
        target_link_libraries(gsplat PUBLIC glm::glm-header-only)
    endif()
    
    set_target_properties(gsplat PROPERTIES LINKER_LANGUAGE CXX)
endif()

# CPU rasterization library
add_library(gsplat_cpu rasterizer/gsplat-cpu/gsplat_cpu.cpp)
target_include_directories(gsplat_cpu PRIVATE ${TORCH_INCLUDE_DIRS})

# Core rendering library source files (removed training-specific files)
set(OPENSPLAT_RENDER_SRC_FILES 
    point_io.cpp
    spherical_harmonics.cpp 
    cv_utils.cpp
    utils.cpp
    project_gaussians.cpp
    rasterize_gaussians.cpp
    rasterize_gaussians_enhanced.cpp
    ssim.cpp
    tensor_math.cpp
    model_render.cpp
    python_bindings.cpp
)

# Optional: Build standalone render library (without Python bindings)
if(BUILD_STANDALONE_LIB)
    add_library(opensplat_render_lib STATIC ${OPENSPLAT_RENDER_SRC_FILES})
    target_include_directories(opensplat_render_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/rasterizer
        ${GPU_INCLUDE_DIRS}
    )
    target_link_libraries(opensplat_render_lib PUBLIC ${STDPPFS_LIBRARY} ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
    target_link_libraries(opensplat_render_lib PRIVATE
        nlohmann_json::nlohmann_json
        nanoflann::nanoflann
    )
    if (NOT WIN32)
        target_link_libraries(opensplat_render_lib PUBLIC pthread)
    endif()
    
    if(GPU_RUNTIME STREQUAL "HIP")
        target_compile_definitions(opensplat_render_lib PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    elseif(GPU_RUNTIME STREQUAL "CUDA")
        target_compile_definitions(opensplat_render_lib PRIVATE USE_CUDA)
    endif()
endif()

add_compile_definitions(GLOG_USE_GLOG_EXPORT)