#include "constants.hpp"
#include "cv_utils_render.hpp"
#include "model_render.hpp"
#include "rasterizer/bindings_render.h"
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;
using namespace torch::indexing;

/**
 * @brief 从字符串解析视图矩阵，支持多种格式
 * @param matrix_str 矩阵字符串，支持空格或逗号分隔
 * @return 一个4x4的torch::Tensor
 */
torch::Tensor parseViewMatrix(const std::string &matrix_str) {
  std::vector<float> values;
  std::istringstream iss(matrix_str);
  std::string token;
  
  // 用空格分隔解析
  while (iss >> token) {
    try {
      values.push_back(std::stof(token));
    } catch (const std::exception &e) {
      throw std::invalid_argument("视图矩阵包含无效数字: " + token);
    }
  }
  
  // 如果空格分隔失败，尝试逗号分隔
  if (values.empty()) {
    std::stringstream ss(matrix_str);
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) {
        try {
          values.push_back(std::stof(token));
        } catch (const std::exception &e) {
          throw std::invalid_argument("视图矩阵包含无效数字: " + token);
        }
      }
    }
  }
  
  if (values.size() != 16) {
    throw std::invalid_argument("视图矩阵必须包含16个元素，当前有 " + std::to_string(values.size()) + " 个元素。");
  }
  
  return torch::from_blob((void *)values.data(), {4, 4}, torch::kFloat32).clone();
}

int main(int argc, char *argv[]) {
  // 1. 解析命令行参数
  cxxopts::Options options("opensplat_render", "3D高斯溅射模型渲染器");

  options.add_options()("h,help", "显示帮助信息")(
      "i,input", "输入PLY文件路径", cxxopts::value<std::string>())(
      "o,output", "输出图像路径",
      cxxopts::value<std::string>()->default_value("output.png"))(
      "d,downscale", "降采样倍数",
      cxxopts::value<float>()->default_value("1.0"))(
      "s,sh-degree", "球谐函数阶数", cxxopts::value<int>()->default_value("3"))(
      "m,view-matrix", "16个元素的视图变换矩阵(行主序)，用空格或逗号分隔",
      cxxopts::value<std::string>())(
      "width", "图像宽度", cxxopts::value<int>()->default_value("800"))(
      "height", "图像高度", cxxopts::value<int>()->default_value("600"))(
      "fx", "焦距fx", cxxopts::value<float>()->default_value("400.0"))(
      "fy", "焦距fy", cxxopts::value<float>()->default_value("400.0"));

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "参数解析错误: " << e.what() << std::endl;
    std::cerr << options.help() << std::endl;
    return EXIT_FAILURE;
  }

  if (result.count("help") || !result.count("input") ||
      !result.count("view-matrix")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // 2. 获取参数
  const std::string ply_path = result["input"].as<std::string>();
  const std::string output_path = result["output"].as<std::string>();
  const float downscale_factor = result["downscale"].as<float>();
  const int sh_degree = result["sh-degree"].as<int>();
  const int width = result["width"].as<int>();
  const int height = result["height"].as<int>();
  const float fx = result["fx"].as<float>();
  const float fy = result["fy"].as<float>();
  const auto view_matrix_str = result["view-matrix"].as<std::string>();

  try {
    // 3. 设置计算设备
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
      std::cout << "使用GPU进行计算" << std::endl;
      device = torch::kCUDA;
    } else {
      std::cout << "使用CPU进行计算" << std::endl;
    }

    // 4. 加载模型
    Model model(sh_degree, device);
    model.loadPly(ply_path);

    // 5. 计算渲染参数
    const float render_fx = fx / downscale_factor;
    const float render_fy = fy / downscale_factor;
    const int render_height =
        static_cast<int>(static_cast<float>(height) / downscale_factor);
    const int render_width =
        static_cast<int>(static_cast<float>(width) / downscale_factor);
    const float render_cx = static_cast<float>(render_width) / 2.0f;
    const float render_cy = static_cast<float>(render_height) / 2.0f;

    // 6. 解析视图矩阵
    torch::Tensor cam_to_world = parseViewMatrix(view_matrix_str).to(device);
    torch::Tensor R = cam_to_world.index({Slice(None, 3), Slice(None, 3)});
    torch::Tensor T = cam_to_world.index({Slice(None, 3), Slice(3, 4)});

    // 调整坐标系以匹配gsplat约定
    R = torch::matmul(
        R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R.device())));

    // 计算世界到相机变换
    torch::Tensor Rinv = R.transpose(0, 1);
    torch::Tensor Tinv = torch::matmul(-Rinv, T);

    torch::Tensor viewMat = torch::eye(4, device);
    viewMat.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
    viewMat.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);

    // 计算投影矩阵
    float fovX =
        2.0f * std::atan(static_cast<float>(render_width) / (2.0f * render_fx));
    float fovY = 2.0f * std::atan(static_cast<float>(render_height) /
                                  (2.0f * render_fy));
    torch::Tensor projMat =
        projectionMatrix(0.001f, 1000.0f, fovX, fovY, device);
    torch::Tensor fullProjMat = torch::matmul(projMat, viewMat);

    // 7. 执行渲染
    std::cout << "开始渲染..." << std::endl;

    // 投影高斯
    torch::Tensor xys, radii, conics, cov2d, camDepths;
    int num_points = model.means.size(0);
    torch::Tensor scales_exp = torch::exp(model.scales);
    torch::Tensor quats_norm = model.quats / model.quats.norm(2, {-1}, true);

    std::tie(xys, radii, conics, cov2d, camDepths) =
        project_gaussians_forward_tensor_cpu(
            num_points, model.means, scales_exp, 1.0f, quats_norm, viewMat,
            fullProjMat, render_fx, render_fy, render_cx, render_cy,
            render_height, render_width, 0.0f);

    if (radii.sum().item<float>() == 0.0f) {
      std::cout << "警告: 没有高斯投影到视锥体内，生成黑色图像" << std::endl;
      cv::Mat image(render_height, render_width, CV_8UC3, cv::Scalar(0, 0, 0));
      cv::imwrite(output_path, image);
      return 0;
    }

    // 计算球谐函数颜色
    torch::Tensor viewDirs = model.means.detach() - T.transpose(0, 1);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    torch::Tensor colors = torch::cat(
        {model.featuresDc.index({Slice(), None, Slice()}), model.featuresRest},
        1);
    torch::Tensor rgbs =
        compute_sh_forward_tensor_cpu(sh_degree, viewDirs, colors);
    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f);

    // 光栅化
    torch::Tensor rgb, final_Ts;
    std::vector<int32_t> *px2gid;
    torch::Tensor opacities_sigmoid = torch::sigmoid(model.opacities);

    std::tie(rgb, final_Ts, px2gid) = rasterize_forward_tensor_cpu(
        render_width, render_height, xys, conics, rgbs, opacities_sigmoid,
        model.backgroundColor, cov2d, camDepths);

    delete[] px2gid;

    // 8. 保存图像
    rgb = torch::clamp_max(rgb, 1.0f);
    cv::Mat image = tensorToImage(rgb.detach().cpu());
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, image);

    std::cout << "渲染完成，图像已保存至: " << output_path << std::endl;
    std::cout << "渲染尺寸: " << render_width << "x" << render_height
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "错误: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}