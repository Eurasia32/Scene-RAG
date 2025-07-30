#ifndef MODEL_RENDER_H
#define MODEL_RENDER_H

#include <torch/torch.h>
#include "spherical_harmonics.hpp"
#include "ssim.hpp"

using namespace torch::indexing;

// Projection matrix utility
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);

// Loss functions (kept for compatibility but can be removed if not needed)
torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt);
torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt);

// Simplified rendering-only model
struct RenderModel {
    torch::Tensor means;        // Gaussian positions [N, 3]
    torch::Tensor scales;       // Gaussian scales [N, 3] 
    torch::Tensor quats;        // Gaussian rotations [N, 4]
    torch::Tensor featuresDc;   // DC spherical harmonics [N, 3]
    torch::Tensor featuresRest; // Higher order SH [N, (deg+1)^2-1, 3]
    torch::Tensor opacities;    // Gaussian opacities [N, 1]
    
    torch::Device device;
    int shDegree;
    
    RenderModel(torch::Tensor means, torch::Tensor scales, torch::Tensor quats,
                torch::Tensor featuresDc, torch::Tensor featuresRest, torch::Tensor opacities,
                const torch::Device &device, int shDegree = 3) :
        means(means), scales(scales), quats(quats), 
        featuresDc(featuresDc), featuresRest(featuresRest), opacities(opacities),
        device(device), shDegree(shDegree) {}
    
    // Render from camera viewpoint
    torch::Tensor render(const torch::Tensor& viewMat, const torch::Tensor& projMat,
                        float fx, float fy, float cx, float cy,
                        int height, int width, const torch::Tensor& background);
};

#endif