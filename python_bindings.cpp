#include "python_bindings.hpp"
#include "project_gaussians.hpp"
#include "rasterize_gaussians_enhanced.hpp"
#include "spherical_harmonics.hpp"
#include "constants.hpp"
#include "utils.hpp"
#include "ply_loader.hpp"
#include "model_render.hpp"
#include <filesystem>

// Forward declaration from model_render.cpp
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);

GaussianRenderer::GaussianRenderer(const std::string& device, int sh_degree)
    : device_(torch::Device(device)), sh_degree_(sh_degree) {
    if (!torch::cuda::is_available() && device == "cuda") {
        throw std::runtime_error("CUDA is not available, use 'cpu' device");
    }
}

GaussianRenderer::~GaussianRenderer() {}

RenderOutput GaussianRenderer::render(const GaussianParams& gaussians,
                                     const CameraParams& camera,
                                     float downsample_factor,
                                     py::object background) {
    torch::Tensor bg;
    if (background.is_none()) {
        bg = torch::zeros({3}, torch::TensorOptions().device(device_));
    } else {
        bg = background.cast<torch::Tensor>().to(device_);
    }
    return render_internal(gaussians, camera, downsample_factor, bg);
}

std::vector<RenderOutput> GaussianRenderer::render_batch(const GaussianParams& gaussians,
                                                        const std::vector<CameraParams>& cameras,
                                                        float downsample_factor,
                                                        py::object background) {
    torch::Tensor bg;
    if (background.is_none()) {
        bg = torch::zeros({3}, torch::TensorOptions().device(device_));
    } else {
        bg = background.cast<torch::Tensor>().to(device_);
    }
    
    std::vector<RenderOutput> outputs;
    outputs.reserve(cameras.size());
    
    for (const auto& camera : cameras) {
        outputs.emplace_back(render_internal(gaussians, camera, downsample_factor, bg));
    }
    
    return outputs;
}

RenderOutput GaussianRenderer::render_internal(const GaussianParams& gaussians,
                                              const CameraParams& camera,
                                              float downsample_factor,
                                              torch::Tensor background) {
    // Ensure all tensors are on the correct device
    torch::Tensor means = gaussians.means.to(device_);
    torch::Tensor scales = gaussians.scales.to(device_);
    torch::Tensor quats = gaussians.quats.to(device_);
    torch::Tensor features_dc = gaussians.features_dc.to(device_);
    torch::Tensor features_rest = gaussians.features_rest.to(device_);
    torch::Tensor opacities = gaussians.opacities.to(device_);
    torch::Tensor world_to_cam = camera.world_to_cam.to(device_);
    background = background.to(device_);
    
    // Apply downsampling
    const float fx = camera.fx / downsample_factor;
    const float fy = camera.fy / downsample_factor;
    const float cx = camera.cx / downsample_factor;
    const float cy = camera.cy / downsample_factor;
    const int height = static_cast<int>(static_cast<float>(camera.height) / downsample_factor);
    const int width = static_cast<int>(static_cast<float>(camera.width) / downsample_factor);
    
    // Extract rotation and translation from world_to_cam matrix
    torch::Tensor R = world_to_cam.index({torch::indexing::Slice(torch::indexing::None, 3), 
                                         torch::indexing::Slice(torch::indexing::None, 3)});
    torch::Tensor T = world_to_cam.index({torch::indexing::Slice(torch::indexing::None, 3), 
                                         torch::indexing::Slice(3, 4)});
    
    // Convert to camera coordinate system (flip y and z for gsplat convention)
    R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, device_)));
    
    // Create view matrix
    torch::Tensor Rinv = R.transpose(0, 1);
    torch::Tensor Tinv = torch::matmul(-Rinv, T);
    torch::Tensor viewMat = torch::eye(4, device_);
    viewMat.index_put_({torch::indexing::Slice(torch::indexing::None, 3), 
                        torch::indexing::Slice(torch::indexing::None, 3)}, Rinv);
    viewMat.index_put_({torch::indexing::Slice(torch::indexing::None, 3), 
                        torch::indexing::Slice(3, 4)}, Tinv);
    
    // Create projection matrix
    float fovX = 2.0f * std::atan(width / (2.0f * fx));
    float fovY = 2.0f * std::atan(height / (2.0f * fy));
    torch::Tensor projMat = projectionMatrix(0.001f, 1000.0f, fovX, fovY, device_);
    
    // Use enhanced rendering function
    EnhancedRenderOutput enhanced_result = render_gaussians_enhanced(
        means, scales, quats, features_dc, features_rest, opacities,
        viewMat, projMat, fx, fy, cx, cy, height, width,
        sh_degree_, background, device_
    );
    
    // Convert px2gid to numpy array
    py::array_t<int32_t> px2gid = convert_px2gid(enhanced_result.px2gid, height, width);
    
    return RenderOutput(torch::clamp_max(enhanced_result.rgb, 1.0f), 
                       enhanced_result.depth, px2gid);
}

GaussianParams GaussianRenderer::load_gaussians(const std::string& ply_path) {
    if (!std::filesystem::exists(ply_path)) {
        throw std::runtime_error("PLY file does not exist: " + ply_path);
    }
    
    // Load point cloud from PLY file
    PointsTensor points = loadPly(ply_path);
    
    // Extract gaussian parameters
    torch::Tensor means = points.xyz;
    torch::Tensor features_dc = points.features_dc;
    torch::Tensor features_rest = points.features_rest;
    torch::Tensor opacities = points.opacities;
    torch::Tensor scales = points.scales;
    torch::Tensor quats = points.quats;
    
    return GaussianParams(means, scales, quats, features_dc, features_rest, opacities);
}

CameraParams GaussianRenderer::create_camera(float fx, float fy, float cx, float cy,
                                           int width, int height,
                                           const std::vector<float>& world_to_cam_matrix) {
    if (world_to_cam_matrix.size() != 16) {
        throw std::runtime_error("world_to_cam_matrix must have 16 elements (4x4 matrix)");
    }
    
    torch::Tensor matrix = torch::from_blob(const_cast<float*>(world_to_cam_matrix.data()), 
                                           {4, 4}, torch::kFloat32).clone();
    
    return CameraParams(fx, fy, cx, cy, width, height, matrix);
}

void GaussianRenderer::set_device(const std::string& device) {
    device_ = torch::Device(device);
}

std::string GaussianRenderer::get_device() const {
    return device_.str();
}

py::array_t<int32_t> GaussianRenderer::convert_px2gid(const torch::Tensor& px2gid_tensor, 
                                                     int height, int width) {
    // px2gid_tensor shape: [height, width, max_gaussians_per_pixel]
    TORCH_CHECK(px2gid_tensor.dim() == 3, "px2gid tensor must be 3-dimensional");
    TORCH_CHECK(px2gid_tensor.size(0) == height && px2gid_tensor.size(1) == width, 
                "px2gid tensor dimensions must match height and width");
    
    int max_gaussians = px2gid_tensor.size(2);
    
    // Create numpy array with same shape
    py::array_t<int32_t> result = py::array_t<int32_t>({height, width, max_gaussians});
    auto result_ptr = static_cast<int32_t*>(result.mutable_unchecked<3>().mutable_data(0, 0, 0));
    
    // Copy data from tensor to numpy array
    torch::Tensor px2gid_cpu = px2gid_tensor.to(torch::kCPU).contiguous();
    const int32_t* tensor_ptr = px2gid_cpu.data_ptr<int32_t>();
    
    std::memcpy(result_ptr, tensor_ptr, height * width * max_gaussians * sizeof(int32_t));
    
    return result;
}