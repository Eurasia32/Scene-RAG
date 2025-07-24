#include "model.hpp"
#include "constants.hpp"
#include <filesystem>

namespace fs = std::filesystem;

// 创建一个OpenGL风格的透视投影矩阵
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY,
                               const torch::Device &device) {
  float t = zNear * std::tan(0.5f * fovY);
  float b = -t;
  float r = zNear * std::tan(0.5f * fovX);
  float l = -r;
  return torch::tensor({{2.0f * zNear / (r - l), 0.0f, (r + l) / (r - l), 0.0f},
                        {0.0f, 2 * zNear / (t - b), (t + b) / (t - b), 0.0f},
                        {0.0f, 0.0f, (zFar + zNear) / (zFar - zNear),
                         -2.0f * zFar * zNear / (zFar - zNear)},
                        {0.0f, 0.0f, 1.0f, 0.0f}},
                       device);
}

// 从PLY文件加载高斯模型 - 简化版本，专门用于渲染
int Model::loadPly(const std::string &filename) {
  std::ifstream f(filename, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("无法打开PLY文件: " + filename);

  std::string line;
  size_t bytesRead = 0;

  // 验证PLY文件头
  std::getline(f, line);
  bytesRead += line.length() + 1;
  if (line.substr(0, 3) != "ply")
    throw std::runtime_error("无效的PLY文件：文件头不是'ply'");

  std::getline(f, line);
  bytesRead += line.length() + 1;
  if (line.find("binary_little_endian 1.0") == std::string::npos)
    throw std::runtime_error("仅支持 'binary_little_endian 1.0' 格式的PLY文件");

  // 读取顶点元素信息
  int numPoints = 0;
  while (std::getline(f, line)) {
    bytesRead += line.length() + 1;
    if (line.find("element vertex") == 0) {
      numPoints = std::stoi(line.substr(15));
    } else if (line.find("end_header") == 0) {
      break;
    }
  }

  if (numPoints == 0)
    throw std::runtime_error("PLY文件头中未找到顶点数量");

  // 重新打开文件并跳过头
  f.close();
  f.open(filename, std::ios::binary);
  f.seekg(bytesRead, std::ios::beg);

  // 计算特征数量
  const int num_sh_features = numShBases(shDegree) - 1;
  const int features_dc_size = 3;
  const int features_rest_size = num_sh_features * 3;

  std::cout << "正在加载 " << numPoints << " 个高斯点..." << std::endl;

  // 为CPU张量分配内存
  torch::Tensor meansCpu = torch::zeros({numPoints, 3}, torch::kFloat32);
  torch::Tensor featuresDcCpu = torch::zeros({numPoints, features_dc_size}, torch::kFloat32);
  torch::Tensor featuresRestCpu = torch::zeros({numPoints, features_rest_size}, torch::kFloat32);
  torch::Tensor opacitiesCpu = torch::zeros({numPoints, 1}, torch::kFloat32);
  torch::Tensor scalesCpu = torch::zeros({numPoints, 3}, torch::kFloat32);
  torch::Tensor quatsCpu = torch::zeros({numPoints, 4}, torch::kFloat32);
  float normals_buffer[3];

  // 读取二进制数据
  for (int i = 0; i < numPoints; ++i) {
    f.read(reinterpret_cast<char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
    f.read(reinterpret_cast<char *>(&normals_buffer), sizeof(float) * 3); // 读取但不使用法线
    f.read(reinterpret_cast<char *>(featuresDcCpu[i].data_ptr()), sizeof(float) * features_dc_size);
    f.read(reinterpret_cast<char *>(featuresRestCpu[i].data_ptr()), sizeof(float) * features_rest_size);
    f.read(reinterpret_cast<char *>(opacitiesCpu[i].data_ptr()), sizeof(float) * 1);
    f.read(reinterpret_cast<char *>(scalesCpu[i].data_ptr()), sizeof(float) * 3);
    f.read(reinterpret_cast<char *>(quatsCpu[i].data_ptr()), sizeof(float) * 4);
  }
  f.close();

  // 将张量移动到目标设备
  means = meansCpu.to(device);
  featuresDc = featuresDcCpu.to(device);
  featuresRest = featuresRestCpu.reshape({numPoints, num_sh_features, 3}).to(device);
  opacities = opacitiesCpu.to(device);
  scales = scalesCpu.to(device);
  quats = quatsCpu.to(device);

  std::cout << "成功加载了 " << means.size(0) << " 个高斯点。" << std::endl;

  return numPoints;
}