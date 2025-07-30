import os
import sys
import torch
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11

def get_extension():
    """Configure the C++ extension."""
    
    # Get PyTorch installation info
    torch_root = torch.utils.cmake_prefix_path
    if isinstance(torch_root, list):
        torch_root = torch_root[0]
    
    # Basic source files for the Python module
    sources = [
        "pybind_module.cpp",
        "python_bindings.cpp", 
        "project_gaussians.cpp",
        "rasterize_gaussians.cpp",
        "spherical_harmonics.cpp",
        "cv_utils.cpp",
        "utils.cpp",
        "tensor_math.cpp",
        "point_io.cpp",
        "ssim.cpp"
    ]
    
    # GPU source files
    gpu_sources = [
        "rasterizer/gsplat/forward.cu",
        "rasterizer/gsplat/backward.cu", 
        "rasterizer/gsplat/bindings.cu",
        "rasterizer/gsplat/ext.cpp"
    ]
    
    # CPU source files  
    cpu_sources = [
        "rasterizer/gsplat-cpu/gsplat_cpu.cpp"
    ]
    
    # Always include CPU sources
    sources.extend(cpu_sources)
    
    # Include directories
    include_dirs = [
        pybind11.get_cmake_dir() + "/../../../include",
        "rasterizer",
        "rasterizer/gsplat",
        "rasterizer/gsplat-cpu"
    ]
    
    # Add PyTorch include directories
    include_dirs.extend(torch.utils.cpp_extension.include_paths())
    
    # Library directories and libraries
    library_dirs = torch.utils.cpp_extension.library_paths()
    libraries = ["torch", "torch_cpu"]
    
    # Compiler and linker flags
    cxx_flags = ["-std=c++17", "-O3"]
    nvcc_flags = []
    
    # Detect GPU support
    cuda_available = torch.cuda.is_available()
    
    # Set up compilation flags based on available hardware
    extra_compile_args = {"cxx": cxx_flags}
    define_macros = []
    
    if cuda_available:
        # Add CUDA sources and configuration
        sources.extend(gpu_sources)
        include_dirs.extend(torch.utils.cpp_extension.include_paths(cuda=True))
        libraries.extend(["torch_cuda", "cuda", "cudart"])
        
        nvcc_flags = [
            "-std=c++17",
            "-O3",
            "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-use_fast_math"
        ]
        
        extra_compile_args["nvcc"] = nvcc_flags
        define_macros.append(("USE_CUDA", None))
        
        # Auto-detect CUDA architectures
        cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if cuda_arch_list is None:
            # Default architectures
            cuda_arch_list = "7.0;7.5;8.0;8.6"
        
        for arch in cuda_arch_list.split(";"):
            if arch:
                arch_flag = f"-gencode=arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}"
                nvcc_flags.append(arch_flag)
    else:
        # CPU-only build
        print("CUDA not available, building CPU-only version")
    
    # Check for external dependencies
    opencv_available = True
    try:
        import cv2
        opencv_include = cv2.includes()
        if opencv_include:
            include_dirs.extend(opencv_include)
    except ImportError:
        opencv_available = False
        print("Warning: OpenCV not found, some features may not work")
    
    if opencv_available:
        libraries.extend(["opencv_core", "opencv_imgproc", "opencv_highgui", "opencv_calib3d"])
    
    # Create extension
    ext = Pybind11Extension(
        "opensplat_render",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language="c++",
        define_macros=define_macros,
        extra_compile_args=extra_compile_args
    )
    
    return ext

# Custom build_ext to handle CUDA compilation
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Check if we're building with CUDA
        if torch.cuda.is_available():
            try:
                # Try to use torch's CUDA extension utilities
                from torch.utils.cpp_extension import CUDAExtension
                print("Building with CUDA support")
            except ImportError:
                print("Warning: torch CUDA extension utilities not available")
        
        super().build_extensions()

setup(
    name="opensplat-render",
    version="1.0.0",
    author="OpenSplat Contributors", 
    author_email="opensplat@example.com",
    description="Python interface for OpenSplat 3D Gaussian Splatting rendering",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=[get_extension()],
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pybind11>=2.6.0"
    ],
    extras_require={
        "opencv": ["opencv-python>=4.5.0"],
        "dev": ["pytest", "black", "isort"]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Computer Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    project_urls={
        "Homepage": "https://github.com/pierotofy/OpenSplat",
        "Bug Reports": "https://github.com/pierotofy/OpenSplat/issues",
        "Source": "https://github.com/pierotofy/OpenSplat"
    }
)