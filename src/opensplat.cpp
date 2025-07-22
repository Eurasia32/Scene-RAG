#include "opensplat.hpp"
#include "cam.hpp"
#include "constants.hpp"
#include "cv_utils.hpp"
#include "utils.hpp"
#include <cxxopts.hpp>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]) {
  cxxopts::Options options(
      "opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
  options.add_options()("i,input", "Path to PLY file",
                        cxxopts::value<std::string>())(
      "o,output", "Path where to save output img",
      cxxopts::value<std::string>()->default_value("splat.png"))(
      "p,plyfile", "PLY file Path",
      cxxopts::value<std::string>()->default_value("splat.ply"))(
      "d,downscale-factor", "Scale input images by this factor.",
      cxxopts::value<float>()->default_value("1"))(
      "sh-degree", "Maximum spherical harmonics degree (must be > 0)",
      cxxopts::value<int>()->default_value("3"));
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << options.help() << std::endl;
    return EXIT_FAILURE;
  }

  const std::string projectRoot = result["input"].as<std::string>();
  const std::string outputScene = result["output"].as<std::string>();
  const std::string plyfile = result["input"].as<std::string>();
  const float downScaleFactor =
      (std::max)(result["downscale-factor"].as<float>(), 1.0f);

  torch::Device device = torch::kCPU;
  int displayStep = 10;

  std::cout << "Using CPU" << std::endl;
  displayStep = 1;

  try {

    Model model();

    model.loadPly(PLYFile);

    cam InputCam;

    const float scaleFactor = downScaleFactor;
    const float fx = InputCam.fx / scaleFactor;
    const float fy = InputCam.fy / scaleFactor;
    const float cx = InputCam.cx / scaleFactor;
    const float cy = InputCam.cy / scaleFactor;
    const int height =
        static_cast<int>(static_cast<float>(InputCam.height) / scaleFactor);
    const int width =
        static_cast<int>(static_cast<float>(InputCam.width) / scaleFactor);

    torch::Tensor R =
        InputCam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
    torch::Tensor T = InputCam.camToWorld.index({Slice(None, 3), Slice(3, 4)});

    // Flip the z and y axes to align with gsplat conventions
    R = torch::matmul(
        R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R.device())));

    // worldToCam
    torch::Tensor Rinv = R.transpose(0, 1);
    torch::Tensor Tinv = torch::matmul(-Rinv, T);

    lastHeight = height;
    lastWidth = width;

    torch::Tensor viewMat = torch::eye(4, device);
    viewMat.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
    viewMat.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);

    float fovX = 2.0f * std::atan(width / (2.0f * fx));
    float fovY = 2.0f * std::atan(height / (2.0f * fy));

    torch::Tensor projMat =
        projectionMatrix(0.001f, 1000.0f, fovX, fovY, device);
    torch::Tensor colors = torch::cat(
        {featuresDc.index({Slice(), None, Slice()}), featuresRest}, 1);

    torch::Tensor conics;
    torch::Tensor cov2d;     // CPU-only
    torch::Tensor camDepths; // CPU-only
    torch::Tensor rgb;

    auto p = ProjectGaussiansCPU::apply(
        means, torch::exp(scales), 1, quats / quats.norm(2, {-1}, true),
        viewMat, torch::matmul(projMat, viewMat), fx, fy, cx, cy, height,
        width);
    xys = p[0];
    radii = p[1];
    conics = p[2];
    cov2d = p[3];
    camDepths = p[4];

    if (radii.sum().item<float>() == 0.0f)
      return backgroundColor.repeat({height, width, 1});

    torch::Tensor viewDirs = means.detach() - T.transpose(0, 1).to(device);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    int degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    torch::Tensor rgbs;

    // std::cout<<viewDirs<<std::endl;

    rgbs = compute_sh_forward_tensor_cpu(degreesToUse, viewDirs, colors);

    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f);

    rgb = RasterizeGaussiansCPU::apply(
        xys, radii, conics, rgbs, torch::sigmoid(opacities), cov2d, camDepths,
        height, width, backgroundColor);

    rgb = torch::clamp_max(rgb, 1.0f);
    cv::Mat image = tensorToImage(rgb.detach().cpu());
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(
        (fs::path(valRender) / (std::to_string(step) + ".png")).string(),
        image);

    catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      exit(1);
    }
  }
}