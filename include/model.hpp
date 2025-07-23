#ifndef MODEL_H
#define MODEL_H

#include "input_data.hpp"
#include <filesystem>
#include <fstream>
#include <gsplat-cpu/bindings.h>
#include <iostream>
#include <torch/torch.h>

namespace fs = std::filesystem;
using namespace torch::indexing;

// 辅助函数：生成一个随机的四元数张量
torch::Tensor randomQuatTensor(long long n);
// 辅助函数：根据相机参数创建投影矩阵
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY,
                               const torch::Device &device);

// Model 结构体现在主要作为高斯模型数据的容器
struct Model {
  // 构造函数：初始化设备和一些默认值
  Model(const InputData &inputData, int shDegree, bool keepCrs,
        const torch::Device &device)
      : shDegree(shDegree), keepCrs(keepCrs), device(device) {

    // 从输入数据中获取坐标系变换信息
    scale = inputData.scale;
    translation = inputData.translation;

    // 初始化背景色
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
  bool keepCrs;
  const torch::Device &device;
  float scale;
  torch::Tensor translation;
};

#endif