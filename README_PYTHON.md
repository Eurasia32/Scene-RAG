# OpenSplat Python Rendering Module

OpenSplat now provides a Python interface for 3D Gaussian Splatting rendering, allowing you to load trained models and render from arbitrary viewpoints with depth information and pixel-to-gaussian correspondence.

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.9+ with CUDA support (optional but recommended)
- OpenCV (optional, for image I/O in examples)

### Install from Source
```bash
# Clone and install in development mode
git clone https://github.com/pierotofy/OpenSplat.git
cd OpenSplat
pip install -e .

# Or install with optional dependencies
pip install -e ".[opencv,dev]"
```

### Use as Submodule
To use OpenSplat as a submodule in your project:

```bash
# In your project directory
git submodule add https://github.com/pierotofy/OpenSplat.git opensplat
cd opensplat
pip install -e .
```

## Quick Start

```python
import opensplat_render as osr
import torch
import numpy as np

# Initialize renderer
renderer = osr.GaussianRenderer(device="cuda", sh_degree=3)

# Load a trained gaussian model
gaussians = osr.GaussianRenderer.load_gaussians("path/to/your/splat.ply")

# Create camera parameters (looking at origin from distance 5)
camera = osr.GaussianRenderer.create_camera(
    fx=400.0, fy=400.0,           # focal length
    cx=400.0, cy=300.0,           # principal point
    width=800, height=600,        # image dimensions
    world_to_cam_matrix=[         # 4x4 transformation matrix (flattened)
        1, 0, 0, 0,
        0, 1, 0, 0, 
        0, 0, 1, 5,
        0, 0, 0, 1
    ]
)

# Render the view
result = renderer.render(
    gaussians=gaussians,
    camera=camera, 
    downsample_factor=1.0,        # render at full resolution
    background=torch.tensor([0.1, 0.2, 0.3])  # background color
)

# Access outputs
rgb_image = result.rgb          # [H, W, 3] RGB values [0-1]
depth_map = result.depth        # [H, W] depth values
px2gid = result.px2gid         # [H, W, max_gaussians] pixel-to-gaussian mapping

print(f"Rendered image shape: {rgb_image.shape}")
print(f"Depth map shape: {depth_map.shape}")
print(f"Pixel-to-Gaussian mapping shape: {px2gid.shape}")
```

## Features

### Multi-View Rendering
```python
# Create multiple camera viewpoints
cameras = []
for angle in np.linspace(0, 2*np.pi, 8):  # 8 views around object
    x = 5 * np.cos(angle)
    z = 5 * np.sin(angle)
    world_to_cam = create_lookat_matrix([x, 0, z], [0, 0, 0], [0, 1, 0])
    
    camera = osr.GaussianRenderer.create_camera(
        400, 400, 400, 300, 800, 600, world_to_cam.flatten()
    )
    cameras.append(camera)

# Batch render all views
results = renderer.render_batch(gaussians, cameras, downsample_factor=2.0)
```

### Depth Analysis
```python
# Get depth statistics
depth = result.depth.cpu().numpy()
print(f"Depth range: {depth.min():.3f} to {depth.max():.3f}")

# Find pixels with specific depth range
near_pixels = np.where((depth > 2.0) & (depth < 4.0))
print(f"Found {len(near_pixels[0])} pixels in depth range [2.0, 4.0]")
```

### Gaussian Analysis
```python
# Find which gaussians contribute to a specific pixel
y, x = 300, 400  # pixel coordinates
contributing_gaussians = px2gid[y, x]  # array of gaussian IDs
valid_gaussians = contributing_gaussians[contributing_gaussians >= 0]

print(f"Pixel ({x}, {y}) has {len(valid_gaussians)} contributing gaussians")
print(f"Gaussian IDs: {valid_gaussians}")

# Get properties of contributing gaussians
if len(valid_gaussians) > 0:
    gaussian_positions = gaussians.means[valid_gaussians]
    gaussian_colors = gaussians.features_dc[valid_gaussians]
    print(f"Contributing gaussian positions: {gaussian_positions}")
```

## Output Formats

### RenderOutput Structure
- **rgb**: `torch.Tensor` of shape `[H, W, 3]` with RGB values in range [0, 1]
- **depth**: `torch.Tensor` of shape `[H, W]` with depth values in world units
- **px2gid**: `numpy.ndarray` of shape `[H, W, max_gaussians_per_pixel]` with gaussian IDs (use -1 for invalid/empty slots)

### Saving Results
```python
import cv2

# Convert and save RGB image
rgb_np = (result.rgb.cpu().numpy() * 255).astype(np.uint8)
rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
cv2.imwrite("rendered.png", rgb_bgr)

# Save depth map
depth_np = result.depth.cpu().numpy()
depth_normalized = ((depth_np - depth_np.min()) / 
                   (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
cv2.imwrite("depth.png", depth_normalized)
```

## GPU Support

The module automatically detects available hardware:

```python
# Check available devices
print("Available devices:", osr.available_devices())

# Use specific device
renderer_cpu = osr.GaussianRenderer(device="cpu")    # CPU-only
renderer_gpu = osr.GaussianRenderer(device="cuda")   # CUDA GPU
```

## Integration Example

```python
class GaussianSplatRenderer:
    def __init__(self, model_path, device="cuda"):
        self.renderer = osr.GaussianRenderer(device=device)
        self.gaussians = osr.GaussianRenderer.load_gaussians(model_path)
        
    def render_trajectory(self, camera_poses, width=800, height=600):
        """Render a camera trajectory."""
        cameras = []
        for pose in camera_poses:
            camera = osr.GaussianRenderer.create_camera(
                fx=0.5*width, fy=0.5*width,  # rough estimate
                cx=width//2, cy=height//2,
                width=width, height=height,
                world_to_cam_matrix=pose.flatten()
            )
            cameras.append(camera)
        
        return self.renderer.render_batch(self.gaussians, cameras)

# Usage
renderer = GaussianSplatRenderer("scene.ply")
results = renderer.render_trajectory(my_camera_poses)
```

## Performance Tips

1. **Use downsampling** for preview/fast rendering: `downsample_factor=2.0` or higher
2. **Batch rendering** is more efficient than individual renders for multiple views
3. **GPU rendering** is significantly faster than CPU when available
4. **Limit SH degree** for faster rendering: `sh_degree=1` instead of `3`

## Troubleshooting

### Build Issues
- Ensure PyTorch is installed with CUDA support matching your system
- Install required build tools: `pip install pybind11 setuptools wheel`
- Check that OpenCV development headers are available if using OpenCV features

### Runtime Issues
- Verify PLY file format matches expected gaussian splatting format
- Check tensor device compatibility (CPU vs CUDA)
- Ensure sufficient GPU memory for large scenes

## License

OpenSplat is licensed under AGPLv3. Commercial use allowed and encouraged under the terms of the license.