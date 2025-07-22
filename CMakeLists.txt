cmake_minimum_required(VERSION 3.21)
project(opensplat)

set(OPENCV_DIR "OPENCV_DIR-NOTFOUND" CACHE PATH "Path to the OPENCV installation directory")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Don't complain about the override from NANOFLANN_BUILD_EXAMPLES
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# Use time-of-extraction for FetchContent'ed files modification time
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_Declare(nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip
)
FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip
)
FetchContent_MakeAvailable(nlohmann_json nanoflann cxxopts)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)

set(STDPPFS_LIBRARY stdc++fs)

find_package(Torch REQUIRED)
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)
find_package(Eigen3 REQUIRED)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)

set(GSPLAT_LIBS gsplat_cpu)

#add_library(gsplat_cpu rasterizer/gsplat-cpu/gsplat_cpu.cpp)
add_library(gsplat_cpu rasterizer/gsplat-cpu/gsplat_cpu.cpp)
target_include_directories(gsplat_cpu PRIVATE ${TORCH_INCLUDE_DIRS})

set(OPENSPLAT_SRC_FILES opensplat.cpp point_io.cpp nerfstudio.cpp model.cpp
kdtree_tensor.cpp cv_utils.cpp utils.cpp project_gaussians.cpp
rasterize_gaussians.cpp ssim.cpp optim_scheduler.cpp input_data.cpp
tensor_math.cpp)

add_executable(opensplat ${OPENSPLAT_SRC_FILES})

install(TARGETS opensplat DESTINATION bin)
set_property(TARGET opensplat PROPERTY CXX_STANDARD 17)
target_include_directories(opensplat PRIVATE
    ${PROJECT_SOURCE_DIR}/rasterizer
)
target_link_libraries(opensplat PUBLIC ${STDPPFS_LIBRARY} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(opensplat PRIVATE
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann
)
target_link_libraries(opensplat PUBLIC pthread)

