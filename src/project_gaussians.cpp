#include "project_gaussians.hpp"

variable_list ProjectGaussiansCPU::apply(
                torch::Tensor means,
                torch::Tensor scales,
                float globScale,
                torch::Tensor quats,
                torch::Tensor viewMat,
                torch::Tensor projMat,
                float fx,
                float fy,
                float cx,
                float cy,
                int imgHeight,
                int imgWidth,
                float clipThresh
            ){
    
    int numPoints = means.size(0);

    auto t = project_gaussians_forward_tensor_cpu(numPoints, means, scales, globScale,
                                              quats, viewMat, projMat, fx, fy,
                                              cx, cy, imgHeight, imgWidth, clipThresh);
                                              
    torch::Tensor xys = std::get<0>(t);
    torch::Tensor radii = std::get<1>(t);
    torch::Tensor conics = std::get<2>(t);
    torch::Tensor cov2d = std::get<3>(t);
    torch::Tensor camDepths = std::get<4>(t);

    return { xys, radii, conics, cov2d, camDepths };
}