#include "gsrender_interface.hpp"
#include "model_render.hpp"
#include "cv_utils_render.hpp"
#include "rasterizer/bindings_render.h"
#include "constants.hpp"

#include <chrono>
#include <iostream>
#include <algorithm>

using namespace torch::indexing;

// GSRenderInterface::Impl - 实现细节类
class GSRenderInterface::Impl {
public:
    std::unique_ptr<Model> model;
    torch::Device device;
    bool model_loaded;
    
    // 背景颜色
    float bg_r, bg_g, bg_b;
    
    Impl() : device(torch::kCPU), model_loaded(false), bg_r(0.0f), bg_g(0.0f), bg_b(0.0f) {}
    
    torch::Tensor parseViewMatrix(const std::vector<float>& matrix_flat) {
        if (matrix_flat.size() != 16) {
            throw std::invalid_argument("视图矩阵必须包含16个元素");
        }
        return torch::from_blob((void*)matrix_flat.data(), {4, 4}, torch::kFloat32).clone().to(device);
    }
    
    std::vector<std::vector<int32_t>> convertPx2gidToVector(const std::vector<int32_t>* px2gid, 
                                                           int width, int height) {
        std::vector<std::vector<int32_t>> result(width * height);
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                size_t pixIdx = i * width + j;
                result[pixIdx] = px2gid[pixIdx];
            }
        }
        
        return result;
    }
};

// GSRenderInterface 实现
GSRenderInterface::GSRenderInterface() : impl_(std::make_unique<Impl>()) {}

GSRenderInterface::~GSRenderInterface() = default;

bool GSRenderInterface::loadModel(const std::string& ply_path, const std::string& device_str) {
    try {
        // 设置设备
        if (device_str == "cuda" && torch::cuda::is_available()) {
            impl_->device = torch::kCUDA;
            std::cout << "使用GPU进行计算" << std::endl;
        } else {
            impl_->device = torch::kCPU;
            std::cout << "使用CPU进行计算" << std::endl;
        }
        
        // 创建和加载模型
        impl_->model = std::make_unique<Model>(3, impl_->device); // 默认sh_degree=3
        impl_->model->loadPly(ply_path);
        impl_->model_loaded = true;
        
        std::cout << "成功加载模型: " << ply_path << std::endl;
        std::cout << "高斯点数量: " << impl_->model->means.size(0) << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "加载模型失败: " << e.what() << std::endl;
        impl_->model_loaded = false;
        return false;
    }
}

RenderResult GSRenderInterface::render(const CameraParams& camera_params,
                                     const std::optional<std::vector<int32_t>>& gaussian_indices) {
    
    if (!impl_->model_loaded) {
        throw std::runtime_error("模型未加载，请先调用loadModel()");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RenderResult result;
    
    try {
        // 计算渲染参数
        const float render_fx = camera_params.fx / camera_params.downscale_factor;
        const float render_fy = camera_params.fy / camera_params.downscale_factor;
        const int render_height = static_cast<int>(camera_params.height / camera_params.downscale_factor);
        const int render_width = static_cast<int>(camera_params.width / camera_params.downscale_factor);
        const float render_cx = camera_params.cx > 0 ? camera_params.cx / camera_params.downscale_factor : render_width / 2.0f;
        const float render_cy = camera_params.cy > 0 ? camera_params.cy / camera_params.downscale_factor : render_height / 2.0f;
        
        result.width = render_width;
        result.height = render_height;
        
        // 解析视图矩阵
        torch::Tensor cam_to_world = camera_params.view_matrix.to(impl_->device);
        torch::Tensor R = cam_to_world.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam_to_world.index({Slice(None, 3), Slice(3, 4)});
        
        // 调整坐标系以匹配gsplat约定
        R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f}, R.device())));
        
        // 计算世界到相机变换
        torch::Tensor Rinv = R.transpose(0, 1);
        torch::Tensor Tinv = torch::matmul(-Rinv, T);
        
        torch::Tensor viewMat = torch::eye(4, impl_->device);
        viewMat.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
        viewMat.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);
        
        // 计算投影矩阵
        float fovX = 2.0f * std::atan(static_cast<float>(render_width) / (2.0f * render_fx));
        float fovY = 2.0f * std::atan(static_cast<float>(render_height) / (2.0f * render_fy));
        torch::Tensor projMat = projectionMatrix(camera_params.near_plane, camera_params.far_plane, fovX, fovY, impl_->device);
        torch::Tensor fullProjMat = torch::matmul(projMat, viewMat);
        
        // 选择要渲染的高斯点
        torch::Tensor means = impl_->model->means;
        torch::Tensor scales = impl_->model->scales;
        torch::Tensor quats = impl_->model->quats;
        torch::Tensor featuresDc = impl_->model->featuresDc;
        torch::Tensor featuresRest = impl_->model->featuresRest;
        torch::Tensor opacities = impl_->model->opacities;
        
        if (gaussian_indices.has_value()) {
            // 过滤高斯点
            torch::Tensor indices_tensor = torch::tensor(*gaussian_indices, torch::kLong).to(impl_->device);
            means = means.index({indices_tensor});
            scales = scales.index({indices_tensor});
            quats = quats.index({indices_tensor});
            featuresDc = featuresDc.index({indices_tensor});
            if (featuresRest.numel() > 0) {
                featuresRest = featuresRest.index({indices_tensor});
            }
            opacities = opacities.index({indices_tensor});
        }
        
        int num_points = means.size(0);
        result.visible_gaussians = num_points;
        
        if (num_points == 0) {
            // 没有高斯点，返回空图像
            result.rgb_image = torch::zeros({render_height, render_width, 3});
            result.depth_image = torch::zeros({render_height, render_width, 1});
            result.final_transmittance = torch::ones({render_height, render_width});
            result.pixel_to_gaussian_mapping.resize(render_width * render_height);
            return result;
        }
        
        // 投影高斯点
        torch::Tensor xys, radii, conics, cov2d, camDepths;
        torch::Tensor scales_exp = torch::exp(scales);
        torch::Tensor quats_norm = quats / quats.norm(2, {-1}, true);
        
        std::tie(xys, radii, conics, cov2d, camDepths) = project_gaussians_forward_tensor_cpu(
            num_points, means, scales_exp, 1.0f, quats_norm, viewMat,
            fullProjMat, render_fx, render_fy, render_cx, render_cy,
            render_height, render_width, 0.0f);
        
        if (radii.sum().item<float>() == 0.0f) {
            std::cout << "警告: 没有高斯投影到视锥体内" << std::endl;
            result.rgb_image = torch::zeros({render_height, render_width, 3});
            result.depth_image = torch::zeros({render_height, render_width, 1});
            result.final_transmittance = torch::ones({render_height, render_width});
            result.pixel_to_gaussian_mapping.resize(render_width * render_height);
            return result;
        }
        
        // 计算球谐函数颜色
        torch::Tensor viewDirs = means.detach() - T.transpose(0, 1);
        viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
        torch::Tensor colors = torch::cat({featuresDc.index({Slice(), None, Slice()}), featuresRest}, 1);
        torch::Tensor rgbs = compute_sh_forward_tensor_cpu(camera_params.sh_degree, viewDirs, colors);
        rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f);
        
        // 光栅化
        torch::Tensor rgb, depth, final_Ts;
        std::vector<int32_t>* px2gid;
        torch::Tensor opacities_sigmoid = torch::sigmoid(opacities);
        
        // 设置背景颜色
        torch::Tensor backgroundColor = torch::tensor({impl_->bg_r, impl_->bg_g, impl_->bg_b}, impl_->device);
        
        std::tie(rgb, depth, final_Ts, px2gid) = rasterize_forward_tensor_cpu(
            render_width, render_height, xys, conics, rgbs, opacities_sigmoid,
            backgroundColor, cov2d, camDepths);
        
        // 转换结果格式
        result.rgb_image = torch::clamp_max(rgb, 1.0f).detach().cpu();
        result.depth_image = depth.detach().cpu().unsqueeze(-1); // 添加channel维度
        result.final_transmittance = final_Ts.detach().cpu();
        
        // 转换px2gid映射
        result.pixel_to_gaussian_mapping = impl_->convertPx2gidToVector(px2gid, render_width, render_height);
        
        // 如果使用了高斯点过滤，需要将局部索引映射回全局索引
        if (gaussian_indices.has_value()) {
            const auto& global_indices = *gaussian_indices;
            for (auto& pixel_gaussians : result.pixel_to_gaussian_mapping) {
                for (auto& local_id : pixel_gaussians) {
                    if (local_id >= 0 && local_id < global_indices.size()) {
                        local_id = global_indices[local_id];
                    }
                }
            }
        }
        
        delete[] px2gid;
        
    } catch (const std::exception& e) {
        std::cerr << "渲染失败: " << e.what() << std::endl;
        throw;
    }
    
    // 计算渲染时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.render_time_ms = duration.count();
    
    return result;
}

std::vector<RenderResult> GSRenderInterface::renderBatch(const std::vector<CameraParams>& camera_params_list,
                                                        const std::optional<std::vector<int32_t>>& gaussian_indices) {
    std::vector<RenderResult> results;
    results.reserve(camera_params_list.size());
    
    for (const auto& params : camera_params_list) {
        results.emplace_back(render(params, gaussian_indices));
    }
    
    return results;
}

GSRenderInterface::ModelInfo GSRenderInterface::getModelInfo() const {
    if (!impl_->model_loaded) {
        throw std::runtime_error("模型未加载");
    }
    
    ModelInfo info;
    info.num_gaussians = impl_->model->means.size(0);
    info.means = impl_->model->means.detach().cpu();
    info.scales = impl_->model->scales.detach().cpu();
    info.rotations = impl_->model->quats.detach().cpu();
    info.opacities = impl_->model->opacities.detach().cpu();
    info.colors_dc = impl_->model->featuresDc.detach().cpu();
    info.colors_rest = impl_->model->featuresRest.detach().cpu();
    info.sh_degree = impl_->model->sh_degree;
    
    return info;
}

std::vector<int32_t> GSRenderInterface::filterGaussians(bool filter_by_position,
                                                       const std::vector<float>& position_bounds,
                                                       bool filter_by_opacity,
                                                       float min_opacity) const {
    if (!impl_->model_loaded) {
        throw std::runtime_error("模型未加载");
    }
    
    std::vector<int32_t> indices;
    int num_gaussians = impl_->model->means.size(0);
    
    torch::Tensor means_cpu = impl_->model->means.cpu();
    torch::Tensor opacities_cpu = torch::sigmoid(impl_->model->opacities).cpu();
    
    for (int i = 0; i < num_gaussians; i++) {
        bool pass_filter = true;
        
        // 位置过滤
        if (filter_by_position && position_bounds.size() >= 6) {
            float x = means_cpu[i][0].item<float>();
            float y = means_cpu[i][1].item<float>();
            float z = means_cpu[i][2].item<float>();
            
            if (x < position_bounds[0] || x > position_bounds[3] ||
                y < position_bounds[1] || y > position_bounds[4] ||
                z < position_bounds[2] || z > position_bounds[5]) {
                pass_filter = false;
            }
        }
        
        // 不透明度过滤
        if (filter_by_opacity && pass_filter) {
            float opacity = opacities_cpu[i].item<float>();
            if (opacity < min_opacity) {
                pass_filter = false;
            }
        }
        
        if (pass_filter) {
            indices.push_back(i);
        }
    }
    
    return indices;
}

void GSRenderInterface::setBackgroundColor(float r, float g, float b) {
    impl_->bg_r = std::clamp(r, 0.0f, 1.0f);
    impl_->bg_g = std::clamp(g, 0.0f, 1.0f);
    impl_->bg_b = std::clamp(b, 0.0f, 1.0f);
}

std::string GSRenderInterface::getDevice() const {
    return impl_->device.is_cuda() ? "cuda" : "cpu";
}

void GSRenderInterface::clearMemory() {
    if (impl_->device.is_cuda()) {
        torch::cuda::empty_cache();
    }
}

void GSRenderInterface::warmup(const CameraParams& camera_params) {
    if (!impl_->model_loaded) return;
    
    std::cout << "预热GPU..." << std::endl;
    
    // 使用较小的分辨率进行预热
    CameraParams warmup_params = camera_params;
    warmup_params.width = 128;
    warmup_params.height = 128;
    
    try {
        render(warmup_params);
        std::cout << "GPU预热完成" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "GPU预热失败: " << e.what() << std::endl;
    }
}

// 工具函数实现
CameraParams createCameraParams(const std::vector<float>& view_matrix_flat,
                               int width, int height,
                               float fx, float fy,
                               float cx, float cy,
                               float downscale_factor,
                               int sh_degree) {
    CameraParams params;
    
    if (view_matrix_flat.size() != 16) {
        throw std::invalid_argument("视图矩阵必须包含16个元素");
    }
    
    params.view_matrix = torch::from_blob((void*)view_matrix_flat.data(), {4, 4}, torch::kFloat32).clone();
    params.width = width;
    params.height = height;
    params.fx = fx;
    params.fy = fy;
    params.cx = cx > 0 ? cx : width / 2.0f;
    params.cy = cy > 0 ? cy : height / 2.0f;
    params.downscale_factor = downscale_factor;
    params.sh_degree = sh_degree;
    
    return params;
}

std::vector<float> tensorToVector(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.cpu().contiguous();
    float* data_ptr = cpu_tensor.data_ptr<float>();
    return std::vector<float>(data_ptr, data_ptr + cpu_tensor.numel());
}

torch::Tensor pixelMappingToTensor(const std::vector<std::vector<int32_t>>& pixel_to_gaussian_mapping,
                                  int width, int height,
                                  int max_gaussians_per_pixel) {
    torch::Tensor result = torch::full({height, width, max_gaussians_per_pixel}, -1, torch::kInt32);
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            size_t pixIdx = i * width + j;
            const auto& pixel_gaussians = pixel_to_gaussian_mapping[pixIdx];
            
            int num_to_copy = std::min((int)pixel_gaussians.size(), max_gaussians_per_pixel);
            for (int k = 0; k < num_to_copy; k++) {
                result[i][j][k] = pixel_gaussians[k];
            }
        }
    }
    
    return result;
}