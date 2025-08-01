import os
import sys
import torch
import subprocess
import tempfile
import shutil
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup, find_packages
import pybind11

def download_and_extract_header_library(name, url, extract_path):
    """Download and extract header-only libraries."""
    import urllib.request
    import zipfile
    
    print(f"Downloading {name} from {url}")
    
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)
        
        with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        os.unlink(tmp_file.name)

def setup_external_dependencies():
    """Setup external header-only dependencies."""
    deps_dir = Path("external_deps")
    deps_dir.mkdir(exist_ok=True)
    
    dependencies = {
        "nlohmann_json": {
            "url": "https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip",
            "include_subdir": "json-3.11.3/single_include"
        },
        "nanoflann": {
            "url": "https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip", 
            "include_subdir": "nanoflann-1.5.5/include"
        },
        "glm": {
            "url": "https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip",
            "include_subdir": "glm-1.0.1"  # This adds glm-1.0.1/ to include path, so #include <glm/glm.hpp> finds glm-1.0.1/glm/glm.hpp
        }
    }
    
    include_paths = []
    
    for name, info in dependencies.items():
        dep_path = deps_dir / name
        if not dep_path.exists():
            print(f"Setting up {name}...")
            download_and_extract_header_library(name, info["url"], deps_dir)
            
        # Find the correct include directory
        include_dir = deps_dir / info["include_subdir"]
        print(f"  Checking include directory: {include_dir}")
        if include_dir.exists():
            include_paths.append(str(include_dir))
            print(f"  -> Added to include paths: {include_dir}")
            # For GLM, verify the header exists
            if name == "glm":
                glm_header = include_dir / "glm" / "glm.hpp"
                print(f"  -> GLM header check: {glm_header} (exists: {glm_header.exists()})")
        else:
            print(f"  Include directory not found: {include_dir}")
            # Fallback: look for common patterns
            extracted_dirs = list(deps_dir.glob(f"{name.replace('_', '-')}*"))
            print(f"  Fallback: found extracted dirs: {extracted_dirs}")
            if extracted_dirs:
                # Try common include subdirectories
                for subdir in ["include", "single_include", ""]:
                    candidate = extracted_dirs[0] / subdir
                    if candidate.exists():
                        include_paths.append(str(candidate))
                        print(f"  -> Fallback added: {candidate}")
                        break
    
    return include_paths

def get_extension():
    """Configure the C++ extension using PyTorch's CUDA extension system."""
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Please install PyTorch with CUDA support.")
    
    # Setup external dependencies (header-only libraries)
    external_includes = setup_external_dependencies()
    
    # Basic source files for the Python module
    sources = [
        "pybind_module.cpp",
        "python_bindings.cpp", 
        "project_gaussians.cpp",
        "rasterize_gaussians.cpp",
        "rasterize_gaussians_enhanced.cpp",
        "spherical_harmonics.cpp",
        "utils.cpp",
        "tensor_math.cpp",
        "point_io.cpp",
        "ssim.cpp",
        "model_render.cpp"
    ]
    
    # GPU source files (always included)
    gpu_sources = [
        "rasterizer/gsplat/forward.cu",
        "rasterizer/gsplat/backward.cu", 
        "rasterizer/gsplat/bindings.cu",
        "rasterizer/gsplat/ext.cpp"
    ]
    
    # Include GPU sources
    sources.extend(gpu_sources)
    
    # Include directories
    include_dirs = [
        ".",  # Current directory
        "rasterizer",
        "rasterizer/gsplat"
    ]
    
    # Add external dependencies include paths
    include_dirs.extend(external_includes)
    
    # Debug: Print all include directories
    print("All include directories:")
    for i, inc_dir in enumerate(include_dirs):
        print(f"  [{i}] {inc_dir}")
        # Check if GLM header exists in this path
        import os
        glm_header = os.path.join(inc_dir, "glm", "glm.hpp")
        if os.path.exists(glm_header):
            print(f"    -> GLM header found: {glm_header}")
    
    # Add pybind11 includes
    include_dirs.append(pybind11.get_include())
    
    # Compiler flags
    extra_compile_args = {
        'cxx': ['-std=c++17', '-O3'],
        'nvcc': [
            '-std=c++17',
            '-O3',
            '--extended-lambda',
            '--expt-relaxed-constexpr',
            '-use_fast_math',
            '-diag-suppress=20012'  # Suppress glm warnings
        ]
    }
    
    # Define macros for CUDA build
    define_macros = [("USE_CUDA", None)]
    
    # Libraries (no OpenCV needed)
    libraries = []
    
    print("Building CUDA-only version")
    
    # Auto-detect CUDA architectures
    cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if cuda_arch_list is None:
        # Default architectures for common GPUs
        cuda_arch_list = ["7.0", "7.5", "8.0", "8.6"]
    else:
        cuda_arch_list = cuda_arch_list.replace(";", " ").split()
    
    # Create CUDA extension
    ext = CUDAExtension(
        name="opensplat_render",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        # PyTorch handles CUDA architectures automatically, but we can specify them
        # if needed via environment variable TORCH_CUDA_ARCH_LIST
    )
    
    return ext

# Custom BuildExtension to handle CUDA compilation and dependencies
class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        # Ensure external dependencies are downloaded before building
        setup_external_dependencies()
        
        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available. Please install PyTorch with CUDA support.")
        
        print("Building with CUDA support")
        super().build_extensions()

    def run(self):
        # Clean up external dependencies on clean build
        if 'clean' in sys.argv:
            external_deps_dir = Path("external_deps")
            if external_deps_dir.exists():
                print("Cleaning external dependencies...")
                shutil.rmtree(external_deps_dir)
        
        super().run()

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