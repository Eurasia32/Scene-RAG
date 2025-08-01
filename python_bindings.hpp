#ifndef PYTHON_BINDINGS_HPP
#define PYTHON_BINDINGS_HPP

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

// Camera parameters structure for Python interface
struct CameraParams {
    float fx, fy, cx, cy;  // Intrinsic parameters
    int width, height;     // Image dimensions
    torch::Tensor world_to_cam;  // 4x4 transformation matrix
    
    CameraParams(float fx, float fy, float cx, float cy, 
                int width, int height, torch::Tensor world_to_cam)
        : fx(fx), fy(fy), cx(cx), cy(cy), width(width), height(height), 
          world_to_cam(world_to_cam) {}
};

// Gaussian model parameters structure
struct GaussianParams {
    torch::Tensor means;        // [N, 3] positions
    torch::Tensor scales;       // [N, 3] scales
    torch::Tensor quats;        // [N, 4] quaternions (rotations)
    torch::Tensor features_dc;  // [N, 3] DC spherical harmonics
    torch::Tensor features_rest;// [N, (deg+1)^2-1, 3] higher order SH
    torch::Tensor opacities;    // [N, 1] opacity values
    
    GaussianParams(torch::Tensor means, torch::Tensor scales, torch::Tensor quats,
                   torch::Tensor features_dc, torch::Tensor features_rest, torch::Tensor opacities)
        : means(means), scales(scales), quats(quats), 
          features_dc(features_dc), features_rest(features_rest), opacities(opacities) {}
};

// Render output structure
struct RenderOutput {
    torch::Tensor rgb;      // [H, W, 3] rendered RGB image
    torch::Tensor depth;    // [H, W] depth map
    py::array_t<int32_t> px2gid;  // [H, W, max_gaussians_per_pixel] pixel to gaussian ID mapping
    
    RenderOutput(torch::Tensor rgb, torch::Tensor depth, py::array_t<int32_t> px2gid)
        : rgb(rgb), depth(depth), px2gid(px2gid) {}
};

// Main rendering functions
class GaussianRenderer {
public:
    GaussianRenderer(const std::string& device = "cuda", int sh_degree = 3);
    ~GaussianRenderer();
    
    // Single view rendering
    RenderOutput render(const GaussianParams& gaussians, 
                       const CameraParams& camera,
                       float downsample_factor = 1.0f,
                       torch::Tensor background = torch::tensor({0.0f, 0.0f, 0.0f}));
    
    // Batch rendering for multiple views
    std::vector<RenderOutput> render_batch(const GaussianParams& gaussians,
                                          const std::vector<CameraParams>& cameras,
                                          float downsample_factor = 1.0f,
                                          torch::Tensor background = torch::tensor({0.0f, 0.0f, 0.0f}));
    
    // Load gaussians from PLY file
    static GaussianParams load_gaussians(const std::string& ply_path);
    
    // Utility functions
    static CameraParams create_camera(float fx, float fy, float cx, float cy,
                                    int width, int height,
                                    const std::vector<float>& world_to_cam_matrix);
    
    void set_device(const std::string& device);
    std::string get_device() const;

private:
    torch::Device device_;
    int sh_degree_;
    
    // Internal rendering implementation
    RenderOutput render_internal(const GaussianParams& gaussians,
                                const CameraParams& camera,
                                float downsample_factor,
                                torch::Tensor background);
    
    // Convert px2gid vector to numpy array
    py::array_t<int32_t> convert_px2gid(const torch::Tensor& px2gid_tensor, int height, int width);
};

#endif // PYTHON_BINDINGS_HPP