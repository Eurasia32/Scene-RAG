# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a specialized **3D Gaussian Splatting (3DGS) rendering system** that has been simplified from a full training+rendering implementation to focus exclusively on high-performance rendering. The project loads PLY files containing trained Gaussian parameters and renders them from arbitrary viewpoints with configurable quality settings.

## Build and Development Commands

### Building the Project
```bash
# Standard build process
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch"
make -j8

# With custom OpenCV path
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch" -DOPENCV_DIR="/path/to/opencv"
```

### Running the Renderer
```bash
# Basic rendering
./opensplat_render -i model.ply -o output.png -m 1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1

# High-resolution with downsampling
./opensplat_render -i model.ply -o render.png -d 2.0 --width 1920 --height 1080 -m [16-element view matrix]

# Use example script for multiple viewpoints
./render_examples.sh model.ply ./output_dir
```

### Development Utilities
```bash
# Clean project and remove redundant files (moves files to backup)
./cleanup_project.sh      # Remove training-related files
./cleanup_rasterizer.sh    # Remove backward propagation code

# Check which files can be safely removed
cat CLEANUP_FILES.md
cat RASTERIZER_CLEANUP.md
```

## Core Architecture

### Rendering Pipeline Flow
1. **Model Loading** → Parse PLY file and load Gaussian parameters to GPU/CPU tensors
2. **View Matrix Processing** → Convert 4x4 camera-to-world matrix to rendering transforms  
3. **Gaussian Projection** → Project 3D ellipsoids to 2D screen space with depth sorting
4. **Spherical Harmonics** → Compute view-dependent colors using SH coefficients
5. **Rasterization** → Alpha-blend overlapped Gaussians with tile-based rendering
6. **Image Output** → Convert tensor to OpenCV Mat with proper color space conversion

### Key Components

**Model Class** (`src/model_render.cpp`, `include/model_render.hpp`):
- Loads PLY files containing trained Gaussian parameters
- Manages core tensors: `means`, `scales`, `quats`, `featuresDc`, `featuresRest`, `opacities`
- Handles device migration (CPU/GPU) automatically

**Rasterization Core** (`src/rasterizer/`):
- `gsplat_render.cpp`: Simplified forward-only rendering implementation
- `bindings_render.h`: Clean API declarations without training functions
- `project_gaussians_forward_tensor_cpu()`: 3D→2D projection with covariance computation
- `compute_sh_forward_tensor_cpu()`: View-dependent color calculation  
- `rasterize_forward_tensor_cpu()`: Final alpha-blending and image generation

**Main Application** (`src/opensplat_render.cpp`):
- Command-line interface with cxxopts
- View matrix parsing and coordinate system conversion
- Device detection and tensor management
- Error handling and progress reporting

### Input/Output Interface

**Required Inputs:**
- PLY file with trained 3DGS model
- 16-element view matrix (row-major, camera-to-world)

**Configurable Parameters:**
- Image resolution and downsampling factor
- Spherical harmonics degree (0-3, affects quality/performance)
- Camera intrinsics (fx, fy focal lengths)

**View Matrix Format:**
The 4x4 transformation matrix uses camera-to-world convention:
```
[R11 R12 R13 Tx]
[R21 R22 R23 Ty] → passed as: R11 R12 R13 Tx R21 R22 R23 Ty R31 R32 R33 Tz 0 0 0 1
[R31 R32 R33 Tz]
[ 0   0   0   1]
```

## Dependencies and Build System

### Core Dependencies (auto-managed by CMake FetchContent):
- **PyTorch C++** (libtorch): Tensor operations and CUDA support
- **OpenCV**: Image I/O and color space conversion  
- **nlohmann/json**: Configuration parsing
- **cxxopts**: Command-line argument processing
- **nanoflann**: Spatial indexing (legacy, minimal usage)

### Build Configuration:
- **C++17** standard required
- **CMake ≥3.21** with FetchContent support
- CUDA auto-detection with CPU fallback
- Release build optimization enabled by default

### Device Support:
The system automatically detects CUDA availability and configures tensors accordingly. All operations work identically on CPU/GPU with performance scaling.

## Performance Considerations

**Quality vs Speed Tradeoffs:**
- Downsampling factor: Higher values = faster rendering, lower resolution
- SH degree: Lower values = faster computation, reduced view-dependent effects
- Image resolution: Directly affects memory usage and rendering time

**Memory Management:**
- Tensors are created on CPU then moved to target device
- Direct tensor operations avoid unnecessary memory copies
- Large models may require GPU memory management

## Coordinate System Conventions

The renderer uses OpenGL-style coordinate systems with Y-up convention. View matrices undergo coordinate flipping (`diag([1, -1, -1])`) to match gsplat's internal conventions.

## Project File Organization

**Active Files (keep these):**
- `src/opensplat_render.cpp` - Main application
- `src/model_render.cpp` - Model loading
- `src/cv_utils_render.cpp` - Image utilities
- `include/*_render.hpp` - Corresponding headers
- `src/rasterizer/gsplat_render.cpp` - Core rendering (forward-only)
- `src/rasterizer/bindings_render.h` - Clean API declarations

**Legacy Files (in backup directories):**
- Original training+rendering implementation
- Backward propagation and gradient computation code
- Point cloud I/O utilities  
- Complex model management for optimization

The project has been deliberately simplified to focus on rendering performance and ease of use, with all training and backward propagation code removed.