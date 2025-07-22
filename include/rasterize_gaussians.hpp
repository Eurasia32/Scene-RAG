#ifndef RASTERIZE_GAUSSIANS_H
#define RASTERIZE_GAUSSIANS_H

#include <torch/torch.h>
#include "tile_bounds.hpp"

using namespace torch::autograd;

class RasterizeGaussiansCPU : public Function<RasterizeGaussiansCPU>{
public:
    static torch::Tensor forward(AutogradContext *ctx, 
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

#endif