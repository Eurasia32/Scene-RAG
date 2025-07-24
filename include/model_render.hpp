#ifndef MODEL_H
#define MODEL_H

#include <bindings_render.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <torch/torch.h>

namespace fs = std::filesystem;
using namespace torch::indexing;

// 辅助函数：根据相机参数创建投影矩阵
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY,
                               const torch::Device &device);

// 简化的Model结构体，专门用于渲染
struct Model {
  // 构造函数：初始化设备和球谐函数阶数
  Model(int shDegree, const torch::Device &device)
      : shDegree(shDegree), device(device) {
    // 初始化背景色为黑色
    backgroundColor = torch::tensor({0.0f, 0.0f, 0.0f}, device);
  }

  // 析构函数
  ~Model() {}

  // 从 PLY 文件加载高斯模型数据
  int loadPly(const std::string &filename);

  // 核心高斯属性
  torch::Tensor means;
  torch::Tensor scales;
  torch::Tensor quats;
  torch::Tensor featuresDc;
  torch::Tensor featuresRest;
  torch::Tensor opacities;
  torch::Tensor backgroundColor;

private:
  // 私有成员变量
  int shDegree;
  const torch::Device &device;
};

#endif