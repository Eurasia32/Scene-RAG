# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **specialized 3D Gaussian Splatting (3DGS) rendering system with Scene-RAG capabilities** that has been streamlined from a full training+rendering implementation to focus exclusively on high-performance rendering and intelligent scene understanding. The project loads PLY files containing trained Gaussian parameters and renders them from arbitrary viewpoints with configurable quality settings, while simultaneously providing semantic segmentation, CLIP feature extraction, and RAG-based scene querying.

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

# Enable Scene-RAG functionality
./opensplat_render -i model.ply -o output.png -m "view_matrix" --enable-rag true --rag-output scene_data

# Scene-RAG with text query
./opensplat_render -i model.ply -o output.png -m "view_matrix" --enable-rag true --rag-query "找到桌子和椅子"

# Export scene graph for analysis
./opensplat_render -i model.ply -o output.png -m "view_matrix" --enable-rag true --export-scene-graph scene_graph.json
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
7. **Scene-RAG Processing** (optional) → Semantic segmentation, CLIP feature extraction, and vector database storage

### Scene-RAG Architecture Flow
```
渲染结果 → 语义分割 → CLIP特征提取 → 向量数据库 → RAG查询
    ↓           ↓            ↓            ↓         ↓
RGB图像    分割掩码     特征向量      聚类存储    智能检索
    ↓           ↓            ↓            ↓         ↓
px2gid映射  边界框信息   512维特征    相似度搜索  结果排序
```

The Scene-RAG system operates by:
1. **Pixel2GaussianMapper**: Uses px2gid data from rasterizer to map 2D pixels to 3D Gaussian points
2. **SegmentationModule**: Performs semantic segmentation (currently K-means, designed for SAM integration)
3. **CLIPFeatureExtractor**: Extracts visual features (currently HSV histograms, designed for CLIP integration)  
4. **VectorDatabase**: Stores and retrieves feature vectors with similarity search
5. **RAGInterface**: Provides natural language and multimodal query capabilities

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
- Scene-RAG integration with conditional processing

**Scene-RAG System** (`src/scene_rag.cpp`, `src/scene_rag_core.cpp`, `include/scene_rag.hpp`):
- **SceneRAG**: Core orchestration class managing the entire pipeline
- **Pixel2GaussianMapper**: Critical component that leverages the `px2gid` array from rasterization to establish 2D-3D correspondences
- **SegmentationModule**: Semantic segmentation (currently simplified K-means, architectured for SAM)
- **CLIPFeatureExtractor**: Visual feature extraction (currently HSV histograms, architectured for CLIP)
- **VectorDatabase**: Feature storage and similarity search with clustering support
- **RAGInterface**: Natural language query interface with multimodal support

### Input/Output Interface

**Required Inputs:**
- PLY file with trained 3DGS model
- 16-element view matrix (row-major, camera-to-world) - **must be quoted**

**Scene-RAG Optional Parameters:**
- `--enable-rag`: Activates semantic processing pipeline
- `--rag-output`: Base path for Scene-RAG data outputs
- `--rag-query`: Natural language query string for semantic search
- `--export-scene-graph`: JSON export path for scene structure

**Configurable Parameters:**
- Image resolution and downsampling factor
- Spherical harmonics degree (0-3, affects quality/performance)
- Camera intrinsics (fx, fy focal lengths)

**Scene-RAG Output Files:**
- `{base}_vectors.db`: Feature vector database in text format
- `{base}_metadata.txt`: Segment metadata including gaussian mappings
- `{base}_mask_{id}.png`: Individual segment masks
- `{base}_query_result_{rank}.png`: Query result visualizations
- Scene graph JSON with segment hierarchy and 3D mappings

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

## Critical Implementation Details

### px2gid Mapping Mechanism
The Scene-RAG system's core innovation is leveraging the `px2gid` array from the rasterization process. This array maps each pixel to the list of Gaussian IDs that contributed to its color, enabling precise 2D-3D correspondence:

```cpp
// In rasterize_forward_tensor_cpu(), each pixel stores contributing Gaussians
px2gid[pixIdx].push_back(gaussianId);

// Scene-RAG uses this to map semantic segments back to 3D points
mapper->getGaussianMapping(segment_mask, gaussian_ids, weights);
```

This mapping is essential because it allows semantic understanding from 2D rendered images to be traced back to specific 3D Gaussian primitives in the scene.

### Coordinate System and Data Flow
The system maintains consistency between rendering coordinates and semantic processing:
- View matrices use camera-to-world convention with OpenGL-style Y-up
- Coordinate flipping (`diag([1, -1, -1])`) applied to match gsplat conventions
- px2gid data preserved until after Scene-RAG processing, then safely deleted
- Semantic segments maintain spatial relationship to original 3D scene structure

## Project File Organization

**Active Files (keep these):**
- `src/opensplat_render.cpp` - Main application with enhanced CLI and Scene-RAG integration
- `src/model_render.cpp` - Simplified model loading
- `src/cv_utils_render.cpp` - Image utilities
- `src/scene_rag.cpp` - Scene-RAG component implementations
- `src/scene_rag_core.cpp` - SceneRAG main class and RAG interface
- `include/scene_rag.hpp` - Complete Scene-RAG system headers
- `include/*_render.hpp` - Corresponding headers
- `src/rasterizer/gsplat_render.cpp` - Core rendering (forward-only) with px2gid output
- `src/rasterizer/bindings_render.h` - Clean API declarations

**Important Data Structures:**
- `SemanticSegment`: Contains mask, features, gaussian mappings, and metadata
- `px2gid` array: Critical bridge between 2D pixels and 3D Gaussian points
- Feature vectors: 512-dimensional (currently simplified, designed for CLIP integration)

**Legacy Files (in backup directories):**
- Original training+rendering implementation
- Backward propagation and gradient computation code
- Point cloud I/O utilities  
- Complex model management for optimization

The project has been deliberately simplified to focus on rendering performance and ease of use, with all training and backward propagation code removed. The Scene-RAG extension adds intelligent scene understanding while maintaining the clean, maintainable codebase optimized for production rendering workloads.

## Common Issues and Solutions

**View Matrix Error**: If you get "视图矩阵必须包含16个元素" error, ensure the matrix string is properly quoted:
```bash
# Wrong (shell splits arguments)
-m 1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1

# Correct (quoted string)
-m "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1"
```

**Scene-RAG Processing Issues**: 
- Ensure sufficient memory for feature extraction and segmentation
- px2gid array is automatically managed - do not manually delete before Scene-RAG processing
- Current segmentation uses K-means clustering; results may vary with scene complexity
- Feature extraction currently uses simplified HSV histograms pending CLIP integration

**Build Issues**: Ensure libtorch path is correctly specified in CMAKE_PREFIX_PATH and all dependencies (OpenCV, Eigen3) are installed.

## Future Extension Points

The Scene-RAG system is architectured for easy enhancement:
- **SegmentationModule**: Drop-in replacement for SAM integration
- **CLIPFeatureExtractor**: Direct CLIP model integration point  
- **VectorDatabase**: FAISS backend integration for large-scale retrieval
- **RAGInterface**: Natural language processing pipeline expansion