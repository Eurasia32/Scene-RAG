#!/usr/bin/env python3
import torch
import opensplat_render as osr
import traceback

def debug_render():
    try:
        print("1. Creating renderer...")
        renderer = osr.GaussianRenderer(device="cuda", sh_degree=3)
        print("   ✓ Renderer created successfully")
        
        print("2. Creating test gaussians...")
        # Create minimal test data
        N = 10  # Just 10 gaussians for testing
        
        means = torch.randn(N, 3, dtype=torch.float32).cuda()
        scales = torch.randn(N, 3, dtype=torch.float32).cuda()  
        quats = torch.randn(N, 4, dtype=torch.float32).cuda()
        quats = quats / quats.norm(dim=1, keepdim=True)  # Normalize
        features_dc = torch.randn(N, 3, dtype=torch.float32).cuda()
        features_rest = torch.randn(N, 15, 3, dtype=torch.float32).cuda()
        opacities = torch.randn(N, 1, dtype=torch.float32).cuda()
        
        gaussians = osr.GaussianParams(means, scales, quats, features_dc, features_rest, opacities)
        print("   ✓ Test gaussians created")
        print(f"   - means shape: {means.shape}, dtype: {means.dtype}")
        print(f"   - features_rest shape: {features_rest.shape}, dtype: {features_rest.dtype}")
        
        print("3. Creating test camera...")
        world_to_cam = [1,0,0,0, 0,1,0,0, 0,0,1,5, 0,0,0,1]  # Identity + translation
        camera = osr.GaussianRenderer.create_camera(400.0, 400.0, 200.0, 150.0, 400, 300, world_to_cam)
        print("   ✓ Test camera created")
        
        print("4. Attempting render...")
        result = renderer.render(gaussians, camera, downsample_factor=1.0)
        print("   ✓ Render successful!")
        print(f"   - RGB shape: {result.rgb.shape}")
        print(f"   - Depth shape: {result.depth.shape}")
        
    except Exception as e:
        print(f"   ✗ Error at step: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to get more specific error info
        print("\nDebug info:")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Caused by: {e.__cause__}")
        
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")

if __name__ == "__main__":
    debug_render()