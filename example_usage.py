#!/usr/bin/env python3
"""
OpenSplat Python Rendering Example

This example demonstrates how to use the opensplat-render Python module
to render Gaussian splats from different camera viewpoints.
"""

import torch
import numpy as np
import opensplat_render as osr

def main():
    print("OpenSplat Python Rendering Example")
    print(f"Version: {osr.version()}")
    print(f"Available devices: {osr.available_devices()}")
    
    # Initialize renderer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    renderer = osr.GaussianRenderer(device=device, sh_degree=3)
    
    # Example 1: Load gaussians from PLY file
    try:
        ply_path = "splat.ply"  # Replace with your PLY file path
        print(f"Loading gaussians from {ply_path}...")
        gaussians = osr.GaussianRenderer.load_gaussians(ply_path)
        print(f"Loaded {gaussians.means.shape[0]} gaussians")
    except:
        # Create dummy gaussians for demonstration
        print("Creating dummy gaussians for demonstration...")
        num_gaussians = 1000
        
        gaussians = osr.GaussianParams(
            means=torch.randn(num_gaussians, 3) * 2.0,  # [N, 3]
            scales=torch.ones(num_gaussians, 3) * 0.1,   # [N, 3] 
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_gaussians),  # [N, 4]
            features_dc=torch.rand(num_gaussians, 3),    # [N, 3]
            features_rest=torch.zeros(num_gaussians, 15, 3),  # [N, 15, 3] for degree 3
            opacities=torch.ones(num_gaussians, 1) * 0.5  # [N, 1]
        )
    
    # Example 2: Create camera parameters
    width, height = 800, 600
    fx = fy = 400.0  # focal length
    cx, cy = width // 2, height // 2  # principal point
    
    # Camera looking at origin from different positions
    positions = [
        [0, 0, 5],   # front
        [3, 0, 3],   # right
        [0, 3, 3],   # top
        [-3, 0, 3],  # left
    ]
    
    cameras = []
    for pos in positions:
        # Create world-to-camera transformation matrix
        # Simple lookAt transformation (camera at pos looking at origin)
        pos = np.array(pos, dtype=np.float32)
        target = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        
        # Compute camera coordinate system
        forward = target - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create rotation matrix (world to camera)
        R = np.array([right, up, -forward])
        
        # Create translation (world to camera)  
        t = -R @ pos
        
        # Combine into 4x4 transformation matrix
        world_to_cam = np.eye(4, dtype=np.float32)
        world_to_cam[:3, :3] = R
        world_to_cam[:3, 3] = t
        
        camera = osr.GaussianRenderer.create_camera(
            fx, fy, cx, cy, width, height, world_to_cam.flatten().tolist()
        )
        cameras.append(camera)
    
    print(f"Created {len(cameras)} camera viewpoints")
    
    # Example 3: Single view rendering
    print("Rendering single view...")
    result = renderer.render(
        gaussians=gaussians,
        camera=cameras[0],
        downsample_factor=1.0,
        background=torch.tensor([0.1, 0.2, 0.3])  # Gray background
    )
    
    print(f"RGB shape: {result.rgb.shape}")
    print(f"Depth shape: {result.depth.shape}")  
    print(f"Pixel-to-Gaussian mapping shape: {result.px2gid.shape}")
    
    # Example 4: Batch rendering
    print("Rendering batch views...")
    batch_results = renderer.render_batch(
        gaussians=gaussians,
        cameras=cameras,
        downsample_factor=2.0  # Render at half resolution for speed
    )
    
    print(f"Rendered {len(batch_results)} views")
    for i, result in enumerate(batch_results):
        print(f"  View {i}: RGB {result.rgb.shape}, Depth {result.depth.shape}")
    
    # Example 5: Save results (requires additional dependencies)
    try:
        import cv2
        print("Saving rendered images...")
        
        for i, result in enumerate(batch_results):
            # Convert RGB to numpy and scale to 0-255
            rgb_np = (result.rgb.cpu().numpy() * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"render_view_{i}.png", rgb_bgr)
            
            # Save depth map
            depth_np = result.depth.cpu().numpy()
            depth_normalized = ((depth_np - depth_np.min()) / 
                              (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
            cv2.imwrite(f"depth_view_{i}.png", depth_normalized)
        
        print("Images saved successfully!")
        
    except ImportError:
        print("OpenCV not available, skipping image saving")
    
    print("Example completed!")


if __name__ == "__main__":
    main()