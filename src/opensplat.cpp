#include "opensplat.hpp"
#include "cam.hpp"
#include "constants.hpp"
#include "cv_utils.hpp"
#include "input_data.hpp"
#include "model.hpp"
#include "point_io.hpp"
#include "tensor_math.hpp"
#include <cxxopts.hpp>
#include <filesystem>
#include <gsplat-cpu/bindings.h>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
using namespace torch::indexing;

/**
 * @brief 从一个包含16个浮点数的vector解析一个4x4的PyTorch张量。
 * @param v 包含16个浮点数的vector，按行主序排列。
 * @return 一个4x4的torch::Tensor。
 */
torch::Tensor parseViewMatrix(const std::vector<float> &v) {
  if (v.size() != 16) {
    throw std::invalid_argument("视图矩阵必须包含16个元素。");
  }
  // 从vector的数据指针创建一个张量，然后克隆它以拥有自己的内存。
  return torch::from_blob((void *)v.data(), {4, 4}, torch::kFloat32).clone();
}

int main(int argc, char *argv[]) {
  // 1. 设置和解析命令行参数
  cxxopts::Options options("opensplat_render",
                           "从给定视点渲染3D高斯溅射模型。");
  options.add_options()("h,help", "打印用法")(
      "i,input-ply", "输入的PLY模型文件路径", cxxopts::value<std::string>())(
      "o,output-image", "保存输出渲染图像的路径",
      cxxopts::value<std::string>()->default_value("render.png"))(
      "d,downscale", "渲染的降采样因子",
      cxxopts::value<float>()->default_value("1.0"))(
      "sh-degree", "用于渲染的球谐函数阶数",
      cxxopts::value<int>()->default_value("3"))(
      "width", "渲染图像宽度", cxxopts::value<int>()->default_value("800"))(
      "height", "渲染图像高度", cxxopts::value<int>()->default_value("600"))(
      "fx", "焦距 fx", cxxopts::value<float>()->default_value("550.0"))(
      "fy", "焦距 fy", cxxopts::value<float>()->default_value("550.0"))(
      "view-matrix", "16个元素的行主序视图矩阵 (camera-to-world)",
      cxxopts::value<std::vector<float>>());

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "参数解析错误: " << e.what() << std::endl;
    std::cerr << options.help() << std::endl;
    return EXIT_FAILURE;
  }

  if (result.count("help") || !result.count("input-ply") ||
      !result.count("view-matrix")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // 2. 从解析结果中获取参数
  const std::string ply_path = result["input-ply"].as<std::string>();
  const std::string output_path = result["output-image"].as<std::string>();
  const float downscale_factor = result["downscale"].as<float>();
  const int sh_degree = result["sh-degree"].as<int>();
  const int width = result["width"].as<int>();
  const int height = result["height"].as<int>();
  const float fx = result["fx"].as<float>();
  const float fy = result["fy"].as<float>();
  const auto view_matrix_vec = result["view-matrix"].as<std::vector<float>>();

  try {
    // 3. 设置计算设备 (CUDA或CPU)
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
      std::cout << "检测到CUDA，使用GPU进行计算。" << std::endl;
      device = torch::kCUDA;
    } else {
      std::cout << "未检测到CUDA，使用CPU进行计算。" << std::endl;
    }

    // 4. 实例化模型并加载PLY文件
    // 创建一个临时的、空的PointSet和InputData，用于实例化Model对象。
    // Model对象将作为高斯数据的容器。
    PointSet *pSet = new PointSet();
    InputData dummy_input_data(pSet, 1.0, torch::zeros({3}));
    delete pSet;

    // 使用虚拟参数实例化模型，因为我们不进行训练。
    Model model(dummy_input_data, 1, 1, 100000, sh_degree, 1000, 100, 100, 1000,
                0.0, 0.0, 100000, 0.0, 100000, false, device);

    std::cout << "正在加载PLY文件: " << ply_path << std::endl;
    model.loadPly(ply_path); // 加载实际的高斯数据，覆盖虚拟数据。
    model.means = model.means.to(device);
    model.scales = model.scales.to(device);
    model.quats = model.quats.to(device);
    model.featuresDc = model.featuresDc.to(device);
    model.featuresRest = model.featuresRest.to(device);
    model.opacities = model.opacities.to(device);
    model.backgroundColor = model.backgroundColor.to(device);

    // 5. 准备渲染所需的矩阵和参数
    const float render_fx = fx / downscale_factor;
    const float render_fy = fy / downscale_factor;
    const int render_height =
        static_cast<int>(static_cast<float>(height) / downscale_factor);
    const int render_width =
        static_cast<int>(static_cast<float>(width) / downscale_factor);
    const float render_cx = static_cast<float>(render_width) / 2.0f;
    const float render_cy = static_cast<float>(render_height) / 2.0f;

    torch::Tensor cam_to_world = parseViewMatrix(view_matrix_vec).to(device);

    torch::Tensor R = cam_to_world.index({Slice(None, 3), Slice(None, 3)});
    torch::Tensor T = cam_to_world.index({Slice(None, 3), Slice(3, 4)});

    // 翻转z和y轴以匹配gsplat的约定
    R = torch::matmul(
        R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R.device())));

    // 计算世界到相机的变换矩阵
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

    // 6. 执行渲染管线
    std::cout << "开始渲染..." << std::endl;

    // 6.1. 投影高斯
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
      std::cout << "警告: 没有高斯被投影到视锥体内。" << std::endl;
      cv::Mat image(render_height, render_width, CV_8UC3,
                    cv::Scalar(model.backgroundColor[2].item<float>() * 255,
                               model.backgroundColor[1].item<float>() * 255,
                               model.backgroundColor[0].item<float>() * 255));
      cv::imwrite(output_path, image);
      return 0;
    }

    // 6.2. 计算球谐函数颜色
    torch::Tensor viewDirs = model.means.detach() - T.transpose(0, 1);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    torch::Tensor colors = torch::cat(
        {model.featuresDc.index({Slice(), None, Slice()}), model.featuresRest},
        1);
    torch::Tensor rgbs =
        compute_sh_forward_tensor_cpu(sh_degree, viewDirs, colors);
    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f);

    // 6.3. 光栅化
    torch::Tensor rgb, final_Ts;
    std::vector<int32_t> *px2gid;
    torch::Tensor opacities_sigmoid = torch::sigmoid(model.opacities);

    std::tie(rgb, final_Ts, px2gid) = rasterize_forward_tensor_cpu(
        render_width, render_height, xys, conics, rgbs, opacities_sigmoid,
        model.backgroundColor, cov2d, camDepths);

    delete[] px2gid; // 释放内存

    // 7. 保存渲染出的图像
    rgb = torch::clamp_max(rgb, 1.0f);
    cv::Mat image = tensorToImage(rgb.detach().cpu());
    // tensorToImage返回的是RGB格式，cv::imwrite需要BGR格式
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(output_path, image);

    std::cout << "渲染完成。图像已保存至: " << output_path << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "运行时发生错误: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}
