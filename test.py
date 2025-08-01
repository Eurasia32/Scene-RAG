import opensplat_render as osr
import torch

# Initialize renderer
renderer = osr.GaussianRenderer(device="cuda", sh_degree=3)

# Load trained gaussians
gaussians = osr.GaussianRenderer.load_gaussians("model.ply")

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