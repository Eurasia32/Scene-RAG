#ifndef CV_UTILS_RENDER
#define CV_UTILS_RENDER

#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// 渲染专用的图像转换函数
cv::Mat tensorToImage(const torch::Tensor &t);

// 深度图转换和保存函数
cv::Mat depthToImage(const torch::Tensor &depth, float min_depth = 0.1f, float max_depth = 100.0f);
void saveDepthImage(const torch::Tensor &depth, const std::string &filename, 
                   float min_depth = 0.1f, float max_depth = 100.0f);

#endif