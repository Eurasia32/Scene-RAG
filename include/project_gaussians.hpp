#ifndef PROJECT_GAUSSIANS_H
#define PROJECT_GAUSSIANS_H

#include <torch/torch.h>
#include "tile_bounds.hpp"
#include <gsplat-cpu/bindings.h>
#include <iostream>

using namespace torch::autograd;

class ProjectGaussiansCPU{
public:
    static variable_list apply( 
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
            float clipThresh = 0.01);
};


#endif