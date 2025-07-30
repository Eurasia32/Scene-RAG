#include "model_render.hpp"
#include "project_gaussians.hpp"
#include "rasterize_gaussians.hpp"
#include "constants.hpp"
#include "tile_bounds.hpp"

torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device){
    // OpenGL perspective projection matrix
    float t = zNear * std::tan(0.5f * fovY);
    float b = -t;
    float r = zNear * std::tan(0.5f * fovX);
    float l = -r;
    return torch::tensor({
        {2.0f * zNear / (r - l), 0.0f, (r + l) / (r - l), 0.0f},
        {0.0f, 2 * zNear / (t - b), (t + b) / (t - b), 0.0f},
        {0.0f, 0.0f, (zFar + zNear) / (zFar - zNear), -1.0f * zFar * zNear / (zFar - zNear)},
        {0.0f, 0.0f, 1.0f, 0.0f}
    }, device);
}

torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt){
    torch::Tensor mse = (rendered - gt).pow(2).mean();
    return (10.f * torch::log10(1.0 / mse));
}

torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt){
    return torch::abs(gt - rendered).mean();
}

torch::Tensor RenderModel::render(const torch::Tensor& viewMat, const torch::Tensor& projMat,
                                 float fx, float fy, float cx, float cy,
                                 int height, int width, const torch::Tensor& background) {
    
    // Combine spherical harmonics features
    torch::Tensor colors = torch::cat({featuresDc.index({Slice(), None, Slice()}), featuresRest}, 1);
    
    // Project gaussians to screen space
    torch::Tensor xys, depths, radii, conics, numTilesHit;
    torch::Tensor cov2d, camDepths; // CPU only
    
    if (device.is_cpu()) {
        auto p = ProjectGaussiansCPU::apply(means,
                                          torch::exp(scales),
                                          1.0f,
                                          quats / quats.norm(2, {-1}, true),
                                          viewMat,
                                          torch::matmul(projMat, viewMat),
                                          fx, fy, cx, cy, height, width);
        xys = p[0];
        radii = p[1];
        conics = p[2];
        cov2d = p[3];
        camDepths = p[4];
    } else {
#if defined(USE_HIP) || defined(USE_CUDA)
        TileBounds tileBounds = std::make_tuple((width + BLOCK_X - 1) / BLOCK_X,
                                              (height + BLOCK_Y - 1) / BLOCK_Y,
                                              1);
        auto p = ProjectGaussians::apply(means,
                                       torch::exp(scales),
                                       1.0f,
                                       quats / quats.norm(2, {-1}, true),
                                       viewMat,
                                       torch::matmul(projMat, viewMat),
                                       fx, fy, cx, cy, height, width,
                                       tileBounds);
        xys = p[0];
        depths = p[1];
        radii = p[2];
        conics = p[3];
        numTilesHit = p[4];
#else
        throw std::runtime_error("GPU support not built, use cpu device");
#endif
    }
    
    // Early exit if no gaussians are visible
    if (radii.sum().item<float>() == 0.0f) {
        return background.repeat({height, width, 1});
    }
    
    // Compute spherical harmonics colors
    torch::Tensor T = viewMat.index({Slice(None, 3), Slice(3, 4)});
    torch::Tensor viewDirs = means.detach() - T.transpose(0, 1).to(device);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    torch::Tensor rgbs;
    
    if (device.is_cpu()) {
        rgbs = SphericalHarmonicsCPU::apply(shDegree, viewDirs, colors);
    } else {
#if defined(USE_HIP) || defined(USE_CUDA)
        rgbs = SphericalHarmonics::apply(shDegree, viewDirs, colors);
#endif
    }
    
    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f);
    
    // Rasterize gaussians
    torch::Tensor rgb;
    
    if (device.is_cpu()) {
        rgb = RasterizeGaussiansCPU::apply(xys, radii, conics, rgbs,
                                          torch::sigmoid(opacities),
                                          cov2d, camDepths,
                                          height, width, background);
    } else {
#if defined(USE_HIP) || defined(USE_CUDA)
        rgb = RasterizeGaussians::apply(xys, depths, radii, conics, numTilesHit,
                                       rgbs, torch::sigmoid(opacities),
                                       height, width, background);
#endif
    }
    
    return torch::clamp_max(rgb, 1.0f);
}