#ifndef CV_UTILS_RENDER
#define CV_UTILS_RENDER

#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// 渲染专用的图像转换函数
cv::Mat tensorToImage(const torch::Tensor &t);

#endif