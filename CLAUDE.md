# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **comprehensive 3D Gaussian Splatting (3DGS) ecosystem** that combines high-performance rendering with intelligent RAG (Retrieval Augmented Generation) capabilities. The project consists of:

1. **Core 3DGS Renderer**: Streamlined C++ rendering engine for PLY models with Python bindings
2. **Intelligent RAG System**: LLM-driven query intent analysis with dynamic gaussian pruning and multi-factor reranking  
3. **Python Integration Layer**: Unified interface connecting CLIP, SAM, and other ML models
4. **Production Examples**: Complete demos and deployment-ready applications

The system enables natural language queries over 3D scenes, with automatic intent understanding, model pruning for performance, and sophisticated result ranking.

## Build and Development Commands

### C++ Rendering Engine
```bash
# Standard build process
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch"
make -j8

# With custom OpenCV path
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch" -DOPENCV_DIR="/path/to/opencv"

# Basic rendering (view matrix must be quoted for proper parsing)
./opensplat_render -i model.ply -o output.png -m "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1"

# High-resolution with depth output
./opensplat_render -i model.ply -o render.png -d 2.0 --width 1920 --height 1080 --depth-output depth.png -m "view_matrix"
```

### Python RAG System
```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -r python/requirements_rag.txt

# Build Python extensions
python setup.py build_ext --inplace

# Run intelligent RAG demos
python examples/dynamic_database_demo.py local           # Use local LLM
python examples/dynamic_database_demo.py openai API_KEY # Use OpenAI GPT
python examples/production_usage.py                     # Production example

# Run basic C++ integration tests
python examples/basic_rendering.py
python examples/clip_sam_integration.py
```

### Development and Testing
```bash
# Clean project and remove redundant files (moves files to backup)
./cleanup_project.sh      # Remove training-related files
./cleanup_rasterizer.sh    # Remove backward propagation code

# Check which files can be safely removed
cat CLEANUP_FILES.md
cat RASTERIZER_CLEANUP.md

# Test intelligent RAG system
python -c "import asyncio; from python.intelligent_rag import quick_search; print(asyncio.run(quick_search('red chair', './model/model.ply')))"
```

## Core Architecture

### Overall System Flow
```
3D Scene (PLY) → C++ Renderer → Python Bindings → RAG System → Natural Language Interface
     ↓              ↓              ↓                ↓                    ↓
  Gaussians    RGB/Depth/px2gid   gsrender     LLM Intent      Query Results
                   Images         module      Analysis         with Ranking
```

### 1. C++ Rendering Pipeline
1. **Model Loading** → Parse PLY file and load Gaussian parameters to GPU/CPU tensors
2. **View Matrix Processing** → Convert 4x4 camera-to-world matrix to rendering transforms  
3. **Gaussian Projection** → Project 3D ellipsoids to 2D screen space with depth sorting
4. **Spherical Harmonics** → Compute view-dependent colors using SH coefficients
5. **Rasterization** → Alpha-blend overlapped Gaussians with tile-based rendering
6. **Image Output** → Convert tensor to OpenCV Mat with proper color space conversion

### 2. Intelligent RAG Pipeline
1. **Query Intent Analysis** → LLM decomposes natural language to structured intent (JSON)
2. **Model Pruning** → Filter gaussians based on spatial/semantic/visual constraints (70%+ reduction)
3. **Initial Retrieval** → Cluster-based search on pruned gaussians with FAISS indexing
4. **Multi-Factor Reranking** → Weighted scoring using 5 similarity factors
5. **Result Synthesis** → Return ranked clusters with confidence scores and metadata

### 3. Python Integration Layer
- **gsrender module**: C++ rendering bindings via pybind11
- **GSRenderRAGBackend**: High-level interface for CLIP/SAM integration
- **IntelligentRAG**: Complete RAG pipeline with caching and performance optimization
- **Production APIs**: Web-ready interfaces with monitoring and error handling

### Key Components

**C++ Core Components**:
- **Model Class** (`src/model_render.cpp`, `include/model_render.hpp`): Simplified PLY loading, manages core tensors (means, scales, quats, featuresDc, featuresRest, opacities), handles device migration (CPU/GPU)
- **Rasterization Core** (`src/rasterizer/gsplat_render.cpp`, `src/rasterizer/bindings_render.h`): Forward-only rendering implementation with 3D→2D projection, spherical harmonics computation, alpha-blending
- **Main Application** (`src/opensplat_render.cpp`): Enhanced CLI with flexible view matrix parsing, device detection, comprehensive error handling
- **Python Bindings** (`src/python_bindings.cpp`, `src/gsrender_interface.cpp`): pybind11 interface exposing C++ functionality to Python

**Python RAG Components**:
- **IntelligentRAG** (`python/intelligent_rag.py`): Complete RAG pipeline with LLM providers (OpenAI, Claude, Local), gaussian pruning, multi-factor reranking
- **GSRenderRAGBackend** (`python/gsrender_rag.py`): High-level interface connecting C++ renderer to CLIP/SAM models, clustering and vector database management
- **Production Examples** (`examples/`): Dynamic database demo, production usage patterns, CLIP/SAM integration examples

**Critical Architectural Decisions**:
- **Simplified from Training**: All backward propagation and training code removed, focus on rendering performance
- **Hybrid C++/Python**: Core rendering in C++ for speed, RAG intelligence in Python for flexibility  
- **Modular LLM Support**: Pluggable LLM providers (OpenAI, Claude, local models) via common interface
- **Intent-Based Pruning**: Dynamic gaussian filtering based on query understanding (2-3x speedup)
- **Multi-Factor Ranking**: Combines vector similarity, text matching, visual attributes, spatial relevance, multi-view consistency

### Input/Output Interface

**C++ Renderer Interface**:
- **Input**: PLY file with trained 3DGS model + 16-element view matrix (row-major, camera-to-world) - **must be quoted**
- **Output**: RGB image, depth map, px2gid mapping (pixel-to-gaussian-ID)
- **Parameters**: Image resolution, downsampling factor, spherical harmonics degree (0-3), camera intrinsics (fx, fy)

**Python RAG Interface**:
```python
# Basic usage
results = await rag.intelligent_search(
    query="red modern chair near window",  # Natural language
    top_k=10,                             # Result count  
    downsample_factor=0.3                 # Speed vs quality
)

# Returns structured results with ranking scores
{
    "final_results": [...],               # Ranked gaussian clusters
    "intent": {...},                      # Parsed query intent
    "performance_metrics": {...},         # Timing and efficiency
    "processing_time": 0.8               # Total time in seconds
}
```

**View Matrix Format** (Critical for C++ renderer):
The 4x4 transformation matrix uses camera-to-world convention and must be quoted:
```bash
# Correct format (quoted string)
-m "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"

# Matrix structure: [R11 R12 R13 Tx R21 R22 R23 Ty R31 R32 R33 Tz 0 0 0 1]
# Wrong: -m 1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1  (shell splits arguments)
```

## Dependencies and Build System

### C++ Dependencies (required for core rendering):
- **PyTorch C++** (libtorch): Tensor operations and CUDA support
- **OpenCV**: Image I/O and color space conversion  
- **Eigen3**: Linear algebra operations

### Python Dependencies (required for RAG system):
```bash
# Core dependencies
pip install torch torchvision numpy scipy scikit-learn faiss-cpu
pip install opencv-python pybind11

# RAG-specific (see python/requirements_rag.txt)
pip install openai anthropic aiohttp              # LLM APIs
pip install matplotlib seaborn tqdm               # Visualization & progress
pip install clip-by-openai segment-anything       # ML models

# Optional performance enhancements  
pip install faiss-gpu cuml cupy                   # GPU acceleration
pip install redis hnswlib                         # Advanced indexing
```

### Auto-managed Dependencies (via CMake FetchContent):
- **nlohmann/json**: Configuration parsing
- **cxxopts**: Command-line argument processing
- **nanoflann**: Spatial indexing (legacy, minimal usage)

### Build Configuration:
- **C++17** standard required
- **CMake ≥3.21** with FetchContent support
- CUDA auto-detection with CPU fallback
- Release build optimization enabled by default
- Python 3.7+ required for Python components

### Device Support:
The system automatically detects CUDA availability and configures tensors accordingly. All operations work identically on CPU/GPU with performance scaling. GPU highly recommended for large models (>100K gaussians).

## Performance Considerations

**Quality vs Speed Tradeoffs**:
- **Downsampling factor**: Higher values = faster rendering, lower resolution
- **SH degree**: Lower values = faster computation, reduced view-dependent effects  
- **Image resolution**: Directly affects memory usage and rendering time
- **RAG pruning factor**: 0.1-0.8 range, lower = faster but may miss results

**Memory Management**:
- **C++ tensors**: Created on CPU then moved to target device, direct operations avoid copies
- **Python caching**: Intent cache (1000 entries) + result cache (500 entries) with TTL
- **Large models**: May require GPU memory management, consider model pruning

**RAG Performance Optimizations**:
- **Intent caching**: Avoids repeated LLM calls for similar queries
- **Gaussian pruning**: 70%+ reduction in processed points with 2-3x speedup
- **Multi-factor ranking**: Parallelized similarity computations
- **Vector indexing**: FAISS/HNSW for sub-linear search complexity

**View Matrix Parsing** (C++ specific):
- Enhanced parser handles various input formats with comprehensive error checking
- Always quote the matrix string to ensure proper shell parsing
- Supports both space and comma-separated values

## Development Status and Production Readiness

**Production-Ready Components** ✅:
- C++ rendering engine (battle-tested, optimized)
- Python bindings and basic integration
- RAG system architecture and framework
- Demo applications and examples

**Prototype/Demo Components** ⚠️ (see TODO_PRODUCTION_READY.md):
- **CLIP feature extraction**: Currently using mock implementation, needs real OpenAI CLIP model
- **3DGS model loading**: Using simulated data, needs actual PLY file parsing  
- **Local LLM integration**: Simplified rule-based fallback, needs Ollama/vLLM/Transformers
- **Semantic labeling**: Position-based inference, needs point cloud segmentation models
- **Vector database**: Basic FAISS, needs optimization for production scale

**Critical Implementation Tasks** (35 total, ~200 hours):
1. **High Priority (15 tasks)**: CLIP integration, PLY parsing, LLM APIs, semantic segmentation, spatial indexing
2. **Medium Priority (12 tasks)**: Multi-view semantics, visual classifiers, performance monitoring, API security  
3. **Low Priority (8 tasks)**: GPU acceleration, distributed deployment, model quantization, A/B testing

## Coordinate System Conventions

The renderer uses OpenGL-style coordinate systems with Y-up convention. View matrices undergo coordinate flipping (`diag([1, -1, -1])`) to match gsplat's internal conventions.

## Project File Organization

**C++ Core (Active Files)**:
- `src/opensplat_render.cpp` - Main application with enhanced CLI
- `src/model_render.cpp` - Simplified model loading
- `src/cv_utils_render.cpp` - Image utilities
- `src/gsrender_interface.cpp` - Python binding interface
- `src/python_bindings.cpp` - pybind11 bindings
- `include/*_render.hpp` - Corresponding headers
- `src/rasterizer/gsplat_render.cpp` - Core rendering (forward-only)
- `src/rasterizer/bindings_render.h` - Clean API declarations

**Python RAG System**:
- `python/intelligent_rag.py` - Complete RAG pipeline with LLM integration (~1300 lines)
- `python/gsrender_rag.py` - High-level CLIP/SAM integration interface
- `examples/dynamic_database_demo.py` - Full system demonstration
- `examples/production_usage.py` - Production deployment patterns
- `examples/clip_sam_integration.py` - ML model integration examples

**Documentation and Guides**:
- `INTELLIGENT_RAG_GUIDE.md` - Comprehensive RAG system usage guide
- `TODO_PRODUCTION_READY.md` - 35-task production readiness checklist
- `PYTHON_RAG_GUIDE.md` - Python backend documentation
- `PYTHON_INTERFACE_GUIDE.md` - C++/Python integration guide

**Legacy Files (in backup directories)**:
- Original training+rendering implementation
- Backward propagation and gradient computation code
- Point cloud I/O utilities  
- Complex model management for optimization

The project has evolved from a pure rendering system to a comprehensive 3D scene understanding platform with intelligent RAG capabilities, while maintaining the clean, optimized C++ core.

## Common Issues and Solutions

**View Matrix Error**: If you get "视图矩阵必须包含16个元素" error, ensure the matrix string is properly quoted:
```bash
# Wrong (shell splits arguments)
-m 1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1

# Correct (quoted string)
-m "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1"
```

**Build Issues**: Ensure libtorch path is correctly specified in CMAKE_PREFIX_PATH and all dependencies (OpenCV, Eigen3) are installed.

**Python Import Errors**: Run `python setup.py build_ext --inplace` to build C++ extensions before importing gsrender module.

**RAG System Issues**: 
- For mock implementations: See TODO_PRODUCTION_READY.md for replacement tasks
- For slow performance: Adjust downsample_factor (0.1-0.3 for speed, 0.5-0.8 for quality)
- For LLM failures: System automatically falls back to rule-based intent analysis

**Memory Issues**: Large models (>1M gaussians) may require GPU memory management or increased RAM. Consider model pruning or distributed processing for production deployment.