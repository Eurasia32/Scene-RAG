#ifndef GSRENDER_INTERFACE_HPP
#define GSRENDER_INTERFACE_HPP

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <optional>

// 前向声明
class Model;

/**
 * @brief 渲染结果结构体，包含RGB、深度和像素-高斯映射
 */
struct RenderResult {
    torch::Tensor rgb_image;        // RGB图像 [H, W, 3]
    torch::Tensor depth_image;      // 深度图像 [H, W, 1] 
    torch::Tensor final_transmittance; // 最终透射率 [H, W]
    
    // px2gid映射数据 - 使用嵌套vector存储每个像素对应的高斯点ID
    std::vector<std::vector<int32_t>> pixel_to_gaussian_mapping;
    
    // 渲染统计信息
    int width, height;
    int visible_gaussians;
    float render_time_ms;
    
    RenderResult() : width(0), height(0), visible_gaussians(0), render_time_ms(0.0f) {}
};

/**
 * @brief 相机参数结构体
 */
struct CameraParams {
    // 相机位姿 (4x4变换矩阵，相机到世界坐标)
    torch::Tensor view_matrix;  // [4, 4]
    
    // 相机内参
    float fx, fy;           // 焦距
    float cx, cy;           // 主点坐标
    
    // 图像参数
    int width, height;      // 图像尺寸
    float downscale_factor; // 降采样倍数
    
    // 渲染参数
    int sh_degree;          // 球谐函数阶数 (0-3)
    float near_plane, far_plane; // 近远平面
    
    CameraParams() 
        : fx(400.0f), fy(400.0f), cx(400.0f), cy(300.0f)
        , width(800), height(600), downscale_factor(1.0f)
        , sh_degree(3), near_plane(0.01f), far_plane(100.0f) {}
};

/**
 * @brief GSRender核心接口类 - 为Python集成优化
 */
class GSRenderInterface {
public:
    GSRenderInterface();
    ~GSRenderInterface();
    
    /**
     * @brief 加载3DGS模型
     * @param ply_path PLY文件路径
     * @param device 计算设备 ("cpu" 或 "cuda")
     * @return 是否加载成功
     */
    bool loadModel(const std::string& ply_path, const std::string& device = "cpu");
    
    /**
     * @brief 渲染场景
     * @param camera_params 相机参数
     * @param gaussian_indices 可选的高斯点索引列表（为空表示使用所有点）
     * @return 渲染结果
     */
    RenderResult render(const CameraParams& camera_params,
                       const std::optional<std::vector<int32_t>>& gaussian_indices = std::nullopt);
    
    /**
     * @brief 批量渲染多个视角
     * @param camera_params_list 相机参数列表
     * @param gaussian_indices 可选的高斯点索引列表
     * @return 渲染结果列表
     */
    std::vector<RenderResult> renderBatch(const std::vector<CameraParams>& camera_params_list,
                                         const std::optional<std::vector<int32_t>>& gaussian_indices = std::nullopt);
    
    /**
     * @brief 获取模型信息
     */
    struct ModelInfo {
        int num_gaussians;
        torch::Tensor means;        // 高斯点位置 [N, 3]
        torch::Tensor scales;       // 缩放参数 [N, 3]  
        torch::Tensor rotations;    // 旋转四元数 [N, 4]
        torch::Tensor opacities;    // 不透明度 [N, 1]
        torch::Tensor colors_dc;    // DC颜色 [N, 3]
        torch::Tensor colors_rest;  // 高阶SH系数 [N, K, 3]
        int sh_degree;
    };
    
    ModelInfo getModelInfo() const;
    
    /**
     * @brief 根据条件过滤高斯点
     * @param filter_by_position 是否按位置过滤
     * @param position_bounds 位置边界 [min_x, min_y, min_z, max_x, max_y, max_z]
     * @param filter_by_opacity 是否按不透明度过滤
     * @param min_opacity 最小不透明度阈值
     * @return 符合条件的高斯点索引列表
     */
    std::vector<int32_t> filterGaussians(bool filter_by_position = false,
                                        const std::vector<float>& position_bounds = {},
                                        bool filter_by_opacity = false, 
                                        float min_opacity = 0.1f) const;
    
    /**
     * @brief 设置背景颜色
     * @param r, g, b 背景颜色 (0-1)
     */
    void setBackgroundColor(float r, float g, float b);
    
    /**
     * @brief 获取设备信息
     */
    std::string getDevice() const;
    
    /**
     * @brief 释放GPU内存
     */
    void clearMemory();
    
    /**
     * @brief 预热GPU（第一次渲染前调用可以提高性能）
     */
    void warmup(const CameraParams& camera_params);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 工具函数：从numpy风格的数组创建相机参数
 */
CameraParams createCameraParams(const std::vector<float>& view_matrix_flat,
                               int width, int height,
                               float fx, float fy,
                               float cx = -1, float cy = -1,
                               float downscale_factor = 1.0f,
                               int sh_degree = 3);

/**
 * @brief 工具函数：将torch tensor转换为numpy兼容格式
 */
std::vector<float> tensorToVector(const torch::Tensor& tensor);

/**
 * @brief 工具函数：将px2gid映射转换为密集tensor格式
 * @param pixel_to_gaussian_mapping 像素到高斯点的映射
 * @param max_gaussians_per_pixel 每个像素最大高斯点数
 * @return 密集tensor [H, W, max_gaussians_per_pixel]，-1表示无效
 */
torch::Tensor pixelMappingToTensor(const std::vector<std::vector<int32_t>>& pixel_to_gaussian_mapping,
                                  int width, int height,
                                  int max_gaussians_per_pixel = 10);

#endif // GSRENDER_INTERFACE_HPP