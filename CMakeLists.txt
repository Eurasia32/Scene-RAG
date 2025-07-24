cmake_minimum_required(VERSION 3.21)
project(opensplat_render)

# 设置OpenCV目录和输出目录
set(OPENCV_DIR "OPENCV_DIR-NOTFOUND" CACHE PATH "Path to the OPENCV installation directory")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 设置 CMake 策略以支持 FetchContent
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# 使用 FetchContent 获取第三方依赖
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

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(STDPPFS_LIBRARY stdc++fs)

# 查找主要依赖包
find_package(Torch REQUIRED)
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)
find_package(Eigen3 REQUIRED)

# 定义链接库
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)
set(GSPLAT_LIBS gsplat_render)

# 创建 gsplat_render 库（简化的渲染专用版本）
add_library(gsplat_render src/rasterizer/gsplat_render.cpp)
target_include_directories(gsplat_render PRIVATE ${TORCH_INCLUDE_DIRS})

# 定义渲染器可执行文件的源文件
set(OPENSPLAT_RENDER_SRC_FILES
    src/opensplat_render.cpp
    src/model_render.cpp
    src/cv_utils_render.cpp
)

# 创建渲染器可执行文件
add_executable(opensplat_render ${OPENSPLAT_RENDER_SRC_FILES})

# 安装目标
install(TARGETS opensplat_render DESTINATION bin)
set_property(TARGET opensplat_render PROPERTY CXX_STANDARD 17)

# 链接库和头文件目录
target_include_directories(opensplat_render PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/rasterizer
)
target_link_libraries(opensplat_render PUBLIC ${STDPPFS_LIBRARY} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(opensplat_render PRIVATE
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann
)
target_link_libraries(opensplat_render PUBLIC pthread)

# 如果需要保留原始的opensplat可执行文件，取消注释以下部分：
# set(OPENSPLAT_ORIGINAL_SRC_FILES
#     src/opensplat.cpp
#     src/point_io.cpp
#     src/model.cpp
#     src/cv_utils.cpp
# )
# add_executable(opensplat ${OPENSPLAT_ORIGINAL_SRC_FILES})
# target_include_directories(opensplat PRIVATE
#     ${PROJECT_SOURCE_DIR}/include
#     ${PROJECT_SOURCE_DIR}/src/rasterizer
# )
# target_link_libraries(opensplat PUBLIC ${STDPPFS_LIBRARY} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(opensplat PRIVATE
#     nlohmann_json::nlohmann_json
#     cxxopts::cxxopts
#     nanoflann::nanoflann
# )
# target_link_libraries(opensplat PUBLIC pthread)