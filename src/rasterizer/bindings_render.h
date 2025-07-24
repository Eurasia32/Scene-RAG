// 3D Gaussian Splatting 渲染核心头文件 - 仅正向传播版本
// 基于 gsplat 项目，专为渲染优化
// Licensed under the AGPLv3

#pragma once

#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/all.h>
#include <tuple>
#include <vector>

// 投影3D高斯到2D屏幕空间
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
project_gaussians_forward_tensor_cpu(
    const int num_points, torch::Tensor &means3d, torch::Tensor &scales,
    const float glob_scale, torch::Tensor &quats, torch::Tensor &viewmat,
    torch::Tensor &projmat, const float fx, const float fy, const float cx,
    const float cy, const unsigned img_height, const unsigned img_width,
    const float clip_thresh);

// 2D光栅化 - Alpha混合，同时输出RGB和深度图
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int32_t> *>
rasterize_forward_tensor_cpu(
    const int width, const int height, const torch::Tensor &xys,
    const torch::Tensor &conics, const torch::Tensor &colors,
    const torch::Tensor &opacities, const torch::Tensor &background,
    const torch::Tensor &cov2d, const torch::Tensor &camDepths);

// 球谐函数基数量
int numShBases(int degree);

// 球谐函数正向计算 - 从视角方向和SH系数计算颜色
torch::Tensor compute_sh_forward_tensor_cpu(const int degrees_to_use,
                                            const torch::Tensor &viewdirs,
                                            const torch::Tensor &coeffs);