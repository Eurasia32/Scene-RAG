#include "cv_utils_render.hpp"

// 将张量转换为OpenCV图像 - 仅保留渲染需要的函数
cv::Mat tensorToImage(const torch::Tensor &t) {
  int h = t.sizes()[0];
  int w = t.sizes()[1];
  int c = t.sizes()[2];

  int type = CV_8UC3;
  if (c != 3)
    throw std::runtime_error("Only images with 3 channels are supported");

  cv::Mat image(h, w, type);
  torch::Tensor scaledTensor = (t * 255.0).toType(torch::kU8);
  uint8_t *dataPtr = static_cast<uint8_t *>(scaledTensor.data_ptr());
  std::copy(dataPtr, dataPtr + (w * h * c), image.data);

  return image;
}

// 将深度张量转换为可视化图像（灰度图）
cv::Mat depthToImage(const torch::Tensor &depth, float min_depth, float max_depth) {
  int h = depth.sizes()[0];
  int w = depth.sizes()[1];

  // 规范化深度值到0-255范围
  torch::Tensor normalized = torch::clamp((depth - min_depth) / (max_depth - min_depth), 0.0f, 1.0f);
  torch::Tensor scaledTensor = (normalized * 255.0).toType(torch::kU8);
  
  cv::Mat depthImage(h, w, CV_8UC1);
  uint8_t *dataPtr = static_cast<uint8_t *>(scaledTensor.data_ptr());
  std::copy(dataPtr, dataPtr + (w * h), depthImage.data);

  return depthImage;
}

// 保存深度图（同时保存原始深度值和可视化图像）
void saveDepthImage(const torch::Tensor &depth, const std::string &filename, 
                   float min_depth, float max_depth) {
  // 保存可视化的深度图（灰度图）
  cv::Mat depthVis = depthToImage(depth, min_depth, max_depth);
  cv::imwrite(filename, depthVis);
  
  // 保存原始深度值（32位浮点数TIFF格式）
  std::string rawFilename = filename;
  size_t dotPos = rawFilename.find_last_of('.');
  if (dotPos != std::string::npos) {
    rawFilename = rawFilename.substr(0, dotPos) + "_raw.tiff";
  } else {
    rawFilename += "_raw.tiff";
  }
  
  int h = depth.sizes()[0];
  int w = depth.sizes()[1];
  cv::Mat rawDepth(h, w, CV_32FC1);
  float *depthPtr = static_cast<float *>(depth.cpu().data_ptr());
  std::copy(depthPtr, depthPtr + (w * h), reinterpret_cast<float*>(rawDepth.data));
  cv::imwrite(rawFilename, rawDepth);
}