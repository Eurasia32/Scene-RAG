# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **specialized 3D Gaussian Splatting (3DGS) rendering system** that has been streamlined from a full training+rendering implementation to focus exclusively on high-performance rendering. The project loads PLY files containing trained Gaussian parameters and renders them from arbitrary viewpoints with configurable quality settings.

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
# Basic rendering (view matrix must be quoted for proper parsing)
./opensplat_render -i model.ply -o output.png -m "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1"

# High-resolution with downsampling
./opensplat_render -i model.ply -o render.png -d 2.0 --width 1920 --height 1080 -m "view_matrix"

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

### Run
./opensplat_render -i ../model/model.ply -m "-0.07069709191336537 -0.7144349248861463 0.6961211527442044 -0.37672130158409256 0.9845301390361363 -0.16213925868582177 -0.06641736310859313 -0.241
32966615991472 0.16031945148508853 0.6806567408729153 0.7148454900045051 1.291332429975879 0 0 0 1" --width 2560 --height 1440 --depth-output depth.png

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
- Simplified PLY loading focused on rendering requirements
- Manages core tensors: `means`, `scales`, `quats`, `featuresDc`, `featuresRest`, `opacities`
- Handles device migration (CPU/GPU) automatically
- Removed all training-related complexity

**Rasterization Core** (`src/rasterizer/`):
- `gsplat_render.cpp`: Simplified forward-only rendering implementation
- `bindings_render.h`: Clean API declarations without training functions
- `project_gaussians_forward_tensor_cpu()`: 3D→2D projection with covariance computation
- `compute_sh_forward_tensor_cpu()`: View-dependent color calculation  
- `rasterize_forward_tensor_cpu()`: Final alpha-blending and image generation

**Main Application** (`src/opensplat_render.cpp`):
- Enhanced command-line interface with flexible view matrix parsing
- Supports both space and comma-separated matrix input formats
- Device detection and tensor management
- Comprehensive error handling and progress reporting

### Input/Output Interface

**Required Inputs:**
- PLY file with trained 3DGS model
- 16-element view matrix (row-major, camera-to-world) - **must be quoted**

**Configurable Parameters:**
- Image resolution and downsampling factor
- Spherical harmonics degree (0-3, affects quality/performance)
- Camera intrinsics (fx, fy focal lengths)

**View Matrix Format:**
The 4x4 transformation matrix uses camera-to-world convention and must be quoted:
```bash
# Correct format (quoted)
-m "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"

# Matrix structure:
# [R11 R12 R13 Tx]
# [R21 R22 R23 Ty] → "R11 R12 R13 Tx R21 R22 R23 Ty R31 R32 R33 Tz 0 0 0 1"
# [R31 R32 R33 Tz]
# [ 0   0   0   1 ]
```

## Dependencies and Build System

### Core Dependencies (must be installed):
- **PyTorch C++** (libtorch): Tensor operations and CUDA support
- **OpenCV**: Image I/O and color space conversion  
- **Eigen3**: Linear algebra operations

### Auto-managed Dependencies (via CMake FetchContent):
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

**View Matrix Parsing:**
- The enhanced parser handles various input formats with comprehensive error checking
- Always quote the matrix string to ensure proper shell parsing
- Supports both space and comma-separated values

## Coordinate System Conventions

The renderer uses OpenGL-style coordinate systems with Y-up convention. View matrices undergo coordinate flipping (`diag([1, -1, -1])`) to match gsplat's internal conventions.

## Project File Organization

**Active Files (keep these):**
- `src/opensplat_render.cpp` - Main application with enhanced CLI
- `src/model_render.cpp` - Simplified model loading
- `src/cv_utils_render.cpp` - Image utilities
- `include/*_render.hpp` - Corresponding headers
- `src/rasterizer/gsplat_render.cpp` - Core rendering (forward-only)
- `src/rasterizer/bindings_render.h` - Clean API declarations

**Legacy Files (in backup directories):**
- Original training+rendering implementation
- Backward propagation and gradient computation code
- Point cloud I/O utilities  
- Complex model management for optimization

The project has been deliberately simplified to focus on rendering performance and ease of use, with all training and backward propagation code removed. This creates a clean, maintainable codebase optimized for production rendering workloads.

## Common Issues and Solutions

**View Matrix Error**: If you get "视图矩阵必须包含16个元素" error, ensure the matrix string is properly quoted:
```bash
# Wrong (shell splits arguments)
-m 1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1

# Correct (quoted string)
-m "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1"
```

**Build Issues**: Ensure libtorch path is correctly specified in CMAKE_PREFIX_PATH and all dependencies (OpenCV, Eigen3) are installed.