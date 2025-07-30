# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Module Installation

OpenSplat now includes a Python rendering interface that can be installed as a pip package:

### Building and Installing
```bash
# Install in development mode
pip install -e .

# Or build wheel and install
pip install .

# Install with optional dependencies
pip install -e ".[opencv,dev]"
```

### Python Usage
```python
import opensplat_render as osr
import torch

# Initialize renderer
renderer = osr.GaussianRenderer(device="cuda", sh_degree=3)

# Load gaussians from PLY file
gaussians = osr.GaussianRenderer.load_gaussians("splat.ply")

# Create camera parameters
camera = osr.GaussianRenderer.create_camera(
    fx=400.0, fy=400.0, cx=400.0, cy=300.0,
    width=800, height=600,
    world_to_cam_matrix=[1,0,0,0, 0,1,0,0, 0,0,1,5, 0,0,0,1]  # 4x4 matrix flattened
)

# Render single view
result = renderer.render(gaussians, camera, downsample_factor=1.0)
print(f"RGB: {result.rgb.shape}")      # [H, W, 3]
print(f"Depth: {result.depth.shape}")  # [H, W]
print(f"PX2GID: {result.px2gid.shape}") # [H, W, max_gaussians_per_pixel]
```

### Key Python Interface Components

**GaussianRenderer Class**: Main rendering interface
- `render()`: Single view rendering with RGB, depth, and pixel-to-gaussian mapping
- `render_batch()`: Batch rendering for multiple viewpoints
- `load_gaussians()`: Load gaussian parameters from PLY files
- `create_camera()`: Utility to create camera parameter structures

**Data Structures**:
- `GaussianParams`: Contains means, scales, quaternions, SH features, opacities
- `CameraParams`: Camera intrinsics and world-to-camera transformation
- `RenderOutput`: RGB image, depth map, and pixel-to-gaussian ID mapping

**Python Module Features**:
- Automatic device detection (CUDA/CPU)
- Support for downsampling during rendering
- Batch processing for multiple camera views
- Native tensor operations through PyTorch integration
- Pixel-to-gaussian correspondence for advanced analysis

### Python-Specific Build Requirements

The Python module requires additional dependencies:
- `pybind11>=2.6.0`: C++ to Python bindings
- `torch>=1.9.0`: PyTorch for tensor operations
- `numpy>=1.19.0`: Numerical operations

Build system automatically detects PyTorch CUDA installation and configures accordingly.

OpenSplat uses CMake for building. The build process varies based on GPU runtime:

### CPU-only Build
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
```

### CUDA Build
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
```

### ROCm/HIP Build
```bash
mkdir build && cd build
export PYTORCH_ROCM_ARCH=gfx906
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME="HIP" -DHIP_ROOT_DIR=/opt/rocm -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON ..
make
```

### macOS with Metal
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME=MPS .. && make -j$(sysctl -n hw.logicalcpu)
```

### Key CMake Options
- `GPU_RUNTIME`: "CUDA", "HIP", "MPS", or "CPU" (default: "CUDA")
- `OPENSPLAT_BUILD_SIMPLE_TRAINER`: Build simple trainer application
- `OPENSPLAT_BUILD_VISUALIZER`: Build visualizer (requires Pangolin)
- `OPENSPLAT_MAX_CUDA_COMPATIBILITY`: Build for maximum CUDA compatibility
- `OPENSPLAT_USE_FAST_MATH`: Enable fast math optimizations

### Running the Program
```bash
# Basic usage
./opensplat /path/to/project -n 2000

# With output file
./opensplat /path/to/project -o output.ply

# Resume training
./opensplat /path/to/project --resume ./splat.ply

# Compressed splat output
./opensplat /path/to/project -o output.splat
```

## Architecture Overview

OpenSplat is a 3D Gaussian Splatting implementation that converts camera poses and sparse points from various photogrammetry formats into 3D scene representations.

### Core Components

**Model (`model.hpp`/`model.cpp`)**: Central training class containing:
- Gaussian parameters (means, scales, quaternions, spherical harmonics, opacities)
- PyTorch optimizers for each parameter type
- Training loop logic with refinement, densification, and pruning
- Forward pass that renders images from camera viewpoints

**Input Data Processing**: Multiple format readers:
- `nerfstudio.hpp`: Nerfstudio project format
- `colmap.hpp`: COLMAP sparse reconstruction
- `opensfm.hpp`: OpenSfM format
- `openmvg.hpp`: OpenMVG format

**Rasterization Engine**: GPU-accelerated rendering in `rasterizer/` directory:
- `gsplat/`: CUDA implementation with forward/backward passes
- `gsplat-cpu/`: CPU fallback implementation  
- `gsplat-metal/`: Apple Metal implementation for macOS

**Key Processing Modules**:
- `project_gaussians.cpp`: Projects 3D Gaussians to 2D screen space
- `rasterize_gaussians.cpp`: Renders Gaussians to images
- `spherical_harmonics.cpp`: Handles view-dependent color representation
- `ssim.cpp`: Structural similarity loss computation
- `optim_scheduler.cpp`: Learning rate scheduling

### Data Flow

1. **Input Processing**: Parse camera poses and sparse points from photogrammetry project
2. **Initialization**: Create initial Gaussian parameters from point cloud
3. **Training Loop**: 
   - Project 3D Gaussians to camera views
   - Rasterize to generate images
   - Compute loss against ground truth images
   - Backpropagate and update Gaussian parameters
   - Periodically refine (split/duplicate/prune) Gaussians based on gradients
4. **Output**: Save trained scene as PLY or compressed SPLAT format

### GPU Runtime Selection

The build system automatically detects and configures for different GPU runtimes:
- **CUDA**: Primary GPU backend for NVIDIA cards
- **HIP**: AMD GPU support via ROCm
- **MPS**: Apple Metal Performance Shaders for macOS
- **CPU**: Fallback CPU implementation

### Dependencies

- **LibTorch**: PyTorch C++ API for tensor operations and automatic differentiation
- **OpenCV**: Image processing and camera calibration utilities
- **External Libraries** (auto-downloaded via CMake):
  - nlohmann/json: JSON parsing for project files
  - nanoflann: KD-tree for spatial queries
  - cxxopts: Command line argument parsing
  - glm: Math library for GPU kernels (CUDA/HIP builds only)

### Training Parameters

Key hyperparameters controlled via command line:
- Resolution scheduling: Start low-res, gradually increase
- Spherical harmonics degree: View-dependent color complexity
- Densification thresholds: When to split/duplicate Gaussians
- Refinement intervals: How often to modify Gaussian population
- Loss weights: Balance between L1 and SSIM losses

The training process uses adaptive Gaussian refinement - splitting large Gaussians with high gradients and duplicating small ones to densify under-represented regions.

## Python Interface Architecture

The Python module extends the original OpenSplat with additional components:

**Enhanced Rasterizer** (`rasterize_gaussians_enhanced.hpp/cpp`): 
- Provides simultaneous RGB, depth, and pixel-to-gaussian mapping output
- CPU implementation with full depth and px2gid support 
- GPU implementation uses existing rasterizer with placeholder depth output

**Python Bindings** (`python_bindings.hpp/cpp`, `pybind_module.cpp`):
- C++ classes wrapped with pybind11 for Python access
- Automatic tensor device management and type conversion
- Memory-safe handling of px2gid arrays

**Build Integration**:
- `setup.py`: PyTorch extension-based build system with CUDA auto-detection
- `pyproject.toml`: Modern Python packaging configuration
- Automatic dependency resolution for different GPU backends

**Key Implementation Details**:
- Pixel-to-gaussian mapping stored as `std::vector<int32_t>*` per pixel, converted to numpy arrays
- Depth maps computed as alpha-weighted average of gaussian depths  
- Background colors handled properly during accumulation
- Memory management ensures proper cleanup of C++ allocated arrays

The Python interface maintains full compatibility with the original training pipeline while providing a clean API for inference and analysis tasks.