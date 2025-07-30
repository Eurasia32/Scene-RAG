# OpenSplat Render - Python 3D Gaussian Splatting Rendering

A lightweight Python interface for 3D Gaussian Splatting rendering, focused on inference and visualization.

## 🎯 Features

- **Fast Rendering**: GPU-accelerated (CUDA/ROCm) and CPU rendering
- **Complete Output**: RGB images, depth maps, and pixel-to-gaussian correspondence
- **Multi-View Support**: Batch rendering for camera trajectories  
- **Easy Integration**: Simple Python API with PyTorch tensors
- **Multiple Backends**: Automatic detection of CUDA, ROCm, or CPU

## 📦 Installation

### From Source
```bash
git clone https://github.com/pierotofy/OpenSplat.git
cd OpenSplat
pip install -e .
```

### As Submodule
```bash
git submodule add https://github.com/pierotofy/OpenSplat.git opensplat
cd opensplat  
pip install -e .
```

### Requirements
- Python 3.7+
- PyTorch 1.9+ (with CUDA for GPU acceleration)
- OpenCV (optional, for examples)

## 🚀 Quick Start

```python
import opensplat_render as osr
import torch

# Initialize renderer
renderer = osr.GaussianRenderer(device="cuda", sh_degree=3)

# Load trained gaussians
gaussians = osr.GaussianRenderer.load_gaussians("scene.ply")

# Create camera
camera = osr.GaussianRenderer.create_camera(
    fx=400.0, fy=400.0, cx=400.0, cy=300.0,
    width=800, height=600,
    world_to_cam_matrix=[1,0,0,0, 0,1,0,0, 0,0,1,5, 0,0,0,1]
)

# Render view
result = renderer.render(gaussians, camera, downsample_factor=1.0)

print(f"RGB: {result.rgb.shape}")      # [H, W, 3] 
print(f"Depth: {result.depth.shape}")  # [H, W]
print(f"PX2GID: {result.px2gid.shape}")# [H, W, max_gaussians]
```

## 📖 API Reference

### GaussianRenderer
```python
renderer = osr.GaussianRenderer(device="cuda", sh_degree=3)
```
- `device`: "cuda", "cpu", or auto-detected
- `sh_degree`: Spherical harmonics degree (1-3)

### Rendering Methods
```python
# Single view
result = renderer.render(gaussians, camera, downsample_factor=1.0, background=torch.tensor([0,0,0]))

# Batch rendering  
results = renderer.render_batch(gaussians, cameras, downsample_factor=2.0)
```

### Data Structures

**GaussianParams**:
- `means`: Positions [N, 3]
- `scales`: Scales [N, 3] 
- `quats`: Rotations [N, 4]
- `features_dc`: DC spherical harmonics [N, 3]
- `features_rest`: Higher order SH [N, (deg+1)²-1, 3]
- `opacities`: Opacity values [N, 1]

**CameraParams**:
- `fx, fy`: Focal length
- `cx, cy`: Principal point
- `width, height`: Image dimensions
- `world_to_cam`: 4×4 transformation matrix

**RenderOutput**:
- `rgb`: RGB image [H, W, 3] in range [0,1]
- `depth`: Depth map [H, W] in world units
- `px2gid`: Pixel-to-gaussian mapping [H, W, max_gaussians]

## 💡 Usage Examples

### Multi-View Rendering
```python
import numpy as np

# Create circular camera trajectory
cameras = []
for angle in np.linspace(0, 2*np.pi, 8):
    x, z = 5*np.cos(angle), 5*np.sin(angle)
    # Create lookAt matrix for camera at [x,0,z] looking at origin
    world_to_cam = create_look_at_matrix([x,0,z], [0,0,0], [0,1,0])
    
    camera = osr.GaussianRenderer.create_camera(
        400, 400, 400, 300, 800, 600, world_to_cam.flatten()
    )
    cameras.append(camera)

# Render all views
results = renderer.render_batch(gaussians, cameras, downsample_factor=2.0)
```

### Depth Analysis  
```python
depth = result.depth.cpu().numpy()
print(f"Depth range: {depth.min():.3f} - {depth.max():.3f}")

# Find objects in specific depth range
near_mask = (depth > 2.0) & (depth < 4.0)
print(f"Pixels in range [2,4]: {near_mask.sum()}")
```

### Gaussian Analysis
```python
# Get gaussians contributing to pixel (x,y)
y, x = 300, 400
gaussian_ids = result.px2gid[y, x]
valid_ids = gaussian_ids[gaussian_ids >= 0]

print(f"Pixel ({x},{y}) has {len(valid_ids)} gaussians")
if len(valid_ids) > 0:
    positions = gaussians.means[valid_ids]
    colors = gaussians.features_dc[valid_ids] 
    print(f"Gaussian positions: {positions}")
```

### Save Results
```python
import cv2

# Save RGB image
rgb_np = (result.rgb.cpu().numpy() * 255).astype(np.uint8)
rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
cv2.imwrite("render.png", rgb_bgr)

# Save depth map
depth_np = result.depth.cpu().numpy()
depth_norm = ((depth_np - depth_np.min()) / 
              (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
cv2.imwrite("depth.png", depth_norm)
```

## ⚙️ Project Structure

```
opensplat/
├── rasterizer/           # GPU/CPU rendering kernels
│   ├── gsplat/          # CUDA kernels  
│   └── gsplat-cpu/      # CPU implementation
├── python_bindings.*    # Python interface
├── pybind_module.cpp    # pybind11 bindings
├── point_io.*           # PLY file loading
├── project_gaussians.*  # 3D→2D projection
├── rasterize_gaussians* # Rendering pipeline
├── spherical_harmonics.*# Color computation
└── setup.py            # pip installation
```

## 📊 Performance Tips

1. **Use GPU**: CUDA/ROCm ~100x faster than CPU
2. **Downsample**: Use `downsample_factor=2.0` for preview
3. **Batch rendering**: More efficient than individual calls
4. **Lower SH degree**: `sh_degree=1` for faster rendering
5. **Limit gaussians**: Large scenes may need memory management

## 🔧 Build Options

```bash
# CPU only
pip install -e . --install-option="--disable-cuda"

# With specific CUDA architectures  
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
pip install -e .

# Debug build
pip install -e . --install-option="--debug"
```

## 🐛 Troubleshooting

**Build Issues**:
- Ensure PyTorch has CUDA support: `torch.cuda.is_available()`
- Install build tools: `pip install pybind11 setuptools wheel`
- Check CUDA toolkit compatibility with PyTorch version

**Runtime Issues**:
- Verify PLY file format (gaussian splatting, not regular point cloud)
- Check tensor devices match (CPU vs CUDA)
- Ensure sufficient GPU memory for large scenes

**Memory Issues**:
- Use downsampling: `downsample_factor=2.0` or higher
- Reduce SH degree: `sh_degree=1`
- Process scenes in chunks for very large models

## 📄 License

AGPLv3 - Commercial use allowed and encouraged under license terms.

## 🤝 Contributing

This is a rendering-focused fork of [OpenSplat](https://github.com/pierotofy/OpenSplat). 
For training functionality, please see the original repository.

---

**Note**: This version removes training capabilities to focus on fast, lightweight rendering. For model training, use the original OpenSplat project.