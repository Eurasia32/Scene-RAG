#ifndef RASTERIZE_GAUSSIANS_ENHANCED_H
#define RASTERIZE_GAUSSIANS_ENHANCED_H

#include <torch/torch.h>
#include <tuple>
#include <vector>
#include "tile_bounds.hpp"
#include "rasterize_gaussians.hpp"

using namespace torch::autograd;

// Enhanced rendering output structure
struct EnhancedRenderOutput {
    torch::Tensor rgb;           // [H, W, 3] RGB image
    torch::Tensor depth;         // [H, W] depth map  
    torch::Tensor alpha;         // [H, W] accumulated alpha
    std::vector<int32_t>* px2gid; // [H*W] pixel to gaussian IDs mapping
};

#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

class RasterizeGaussiansEnhanced : public Function<RasterizeGaussiansEnhanced>{
public:
    static EnhancedRenderOutput forward(AutogradContext *ctx, 
            torch::Tensor xys,
            torch::Tensor depths,
            torch::Tensor radii,
            torch::Tensor conics,
            torch::Tensor numTilesHit,
            torch::Tensor colors,
            torch::Tensor opacity,
            int imgHeight,
            int imgWidth,
            torch::Tensor background);
    
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

#endif

class RasterizeGaussiansCPUEnhanced : public Function<RasterizeGaussiansCPUEnhanced>{
public:
    static EnhancedRenderOutput forward(AutogradContext *ctx, 
            torch::Tensor xys,
            torch::Tensor radii,
            torch::Tensor conics,
            torch::Tensor colors,
            torch::Tensor opacity,
            torch::Tensor cov2d,
            torch::Tensor camDepths,
            int imgHeight,
            int imgWidth,
            torch::Tensor background);
    
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

// Utility functions for enhanced rendering
EnhancedRenderOutput render_gaussians_enhanced(
    torch::Tensor means,
    torch::Tensor scales,
    torch::Tensor quats,
    torch::Tensor features_dc,
    torch::Tensor features_rest,
    torch::Tensor opacities,
    torch::Tensor viewMat,
    torch::Tensor projMat,
    float fx, float fy, float cx, float cy,
    int height, int width,
    int sh_degree,
    torch::Tensor background,
    const torch::Device& device
);

#endif // RASTERIZE_GAUSSIANS_ENHANCED_H