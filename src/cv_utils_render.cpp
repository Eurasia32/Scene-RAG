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