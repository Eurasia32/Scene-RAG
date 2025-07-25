from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils import cpp_extension
import glob
import os

# 检查PyTorch版本和CUDA支持
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {'是' if torch.cuda.is_available() else '否'}")

# 源文件列表
cpp_sources = [
    "src/python_bindings.cpp",
    "src/gsrender_interface.cpp", 
    "src/model_render.cpp",
    "src/cv_utils_render.cpp",
    "src/rasterizer/gsplat_render.cpp"
]

# 头文件目录
include_dirs = [
    "include",
    "src/rasterizer",
    pybind11.get_include(),
] + cpp_extension.include_paths()

# 库目录
library_dirs = cpp_extension.library_paths()

# 链接库
libraries = ["torch", "torch_cpu"]
if torch.cuda.is_available():
    libraries.append("torch_cuda")

# 编译器参数
cxx_flags = [
    "-std=c++17",
    "-O3",
    "-DWITH_OPENCV"
]

# 链接器参数  
link_flags = []

# 检查OpenCV
try:
    import cv2
    opencv_include = cv2.includes()
    if opencv_include:
        include_dirs.extend(opencv_include)
    # 常见OpenCV库路径
    opencv_lib_paths = [
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu", 
        "/opt/opencv/lib"
    ]
    for path in opencv_lib_paths:
        if os.path.exists(path):
            library_dirs.append(path)
    
    libraries.extend(["opencv_core", "opencv_imgproc", "opencv_highgui"])
    print("OpenCV支持: 已启用")
except ImportError:
    print("OpenCV支持: 未找到")

# 检查Eigen3
eigen_paths = [
    "/usr/include/eigen3",
    "/usr/local/include/eigen3",
    "/opt/eigen3/include"
]
for path in eigen_paths:
    if os.path.exists(path):
        include_dirs.append(path)
        print(f"Eigen3支持: {path}")
        break

# CUDA编译支持
if torch.cuda.is_available():
    print("启用CUDA编译支持")
    cxx_flags.append("-DWITH_CUDA")

# 创建扩展模块
ext_modules = [
    Pybind11Extension(
        "gsrender",
        sources=cpp_sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        cxx_std=17,
        extra_compile_args=cxx_flags,
        extra_link_args=link_flags,
    ),
]

# 自定义构建类
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # 检查编译器
        if self.compiler.compiler_type == 'unix':
            # 添加PyTorch编译标志
            for ext in self.extensions:
                ext.extra_compile_args.extend(cpp_extension.COMMON_NVCC_FLAGS)
        
        super().build_extensions()

if __name__ == "__main__":
    from setuptools import setup
    
    setup(
        name="gsrender",
        version="1.0.0", 
        author="GSRender Team",
        author_email="",
        description="3D Gaussian Splatting渲染接口，支持CLIP和SAM集成",
        long_description=open("README.md").read() if os.path.exists("README.md") else "",
        long_description_content_type="text/markdown",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=[
            "numpy>=1.19.0",
            "torch>=1.9.0", 
            "opencv-python>=4.5.0",
            "pybind11>=2.6.0"
        ],
        extras_require={
            "clip": ["clip-by-openai", "Pillow"],
            "sam": ["segment-anything", "opencv-python"],
            "dev": ["pytest", "black", "flake8"]
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research", 
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Processing",
            "License :: OSI Approved :: GNU Affero General Public License v3",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8", 
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: C++",
        ],
        project_urls={
            "Bug Reports": "https://github.com/your-repo/issues",
            "Source": "https://github.com/your-repo", 
        },
    )