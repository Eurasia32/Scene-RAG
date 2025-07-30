#include "rasterize_gaussians_enhanced.hpp"
#include "project_gaussians.hpp"
#include "spherical_harmonics.hpp"
#include "gsplat.hpp"
#include "constants.hpp"

EnhancedRenderOutput RasterizeGaussiansCPUEnhanced::forward(AutogradContext *ctx, 
        torch::Tensor xys,
        torch::Tensor radii,
        torch::Tensor conics,
        torch::Tensor colors,
        torch::Tensor opacity,
        torch::Tensor cov2d,
        torch::Tensor camDepths,
        int imgHeight,
        int imgWidth,
        torch::Tensor background) {
    
    int channels = colors.size(1);
    int numPoints = xys.size(0);
    int width = imgWidth;
    int height = imgHeight;
    
    float *pDepths = static_cast<float *>(camDepths.data_ptr());
    std::vector<int32_t> *px2gid = new std::vector<int32_t>[width * height];
    
    // Sort gaussians by depth
    std::vector<size_t> gIndices(numPoints);
    std::iota(gIndices.begin(), gIndices.end(), 0);
    std::sort(gIndices.begin(), gIndices.end(), [&pDepths](int a, int b){
        return pDepths[a] < pDepths[b];
    });
    
    torch::Device device = xys.device();
    
    // Initialize output tensors
    torch::Tensor outImg = torch::zeros({height, width, channels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor depthMap = torch::zeros({height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor alphaMap = torch::zeros({height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor finalTs = torch::ones({height, width}, torch::TensorOptions().dtype(torch::kFloat32).device(device));   
    torch::Tensor done = torch::zeros({height, width}, torch::TensorOptions().dtype(torch::kBool).device(device));   

    torch::Tensor sqCov2dX = 3.0f * torch::sqrt(cov2d.index({"...", 0, 0}));
    torch::Tensor sqCov2dY = 3.0f * torch::sqrt(cov2d.index({"...", 1, 1}));
    
    // Get pointers to tensor data
    float *pConics = static_cast<float *>(conics.data_ptr());
    float *pCenters = static_cast<float *>(xys.data_ptr());
    float *pSqCov2dX = static_cast<float *>(sqCov2dX.data_ptr());
    float *pSqCov2dY = static_cast<float *>(sqCov2dY.data_ptr());
    float *pOpacities = static_cast<float *>(opacity.data_ptr());
    float *pColors = static_cast<float *>(colors.data_ptr());
    float *pBg = static_cast<float *>(background.data_ptr());

    float *pOutImg = static_cast<float *>(outImg.data_ptr());
    float *pDepthMap = static_cast<float *>(depthMap.data_ptr());
    float *pAlphaMap = static_cast<float *>(alphaMap.data_ptr());
    float *pFinalTs = static_cast<float *>(finalTs.data_ptr());
    bool *pDone = static_cast<bool *>(done.data_ptr());
    
    // Render each gaussian (back to front)
    for (const size_t &gaussianIdx : gIndices){
        const float opacity = pOpacities[gaussianIdx];
        if (opacity < 1.0f / 255.0f) continue;
        
        const float centerX = pCenters[gaussianIdx * 2 + 0];
        const float centerY = pCenters[gaussianIdx * 2 + 1];
        const float sqCovX = pSqCov2dX[gaussianIdx];
        const float sqCovY = pSqCov2dY[gaussianIdx];
        
        // Gaussian bounding box
        const int minX = static_cast<int>(std::max(0.0f, centerX - sqCovX));
        const int maxX = static_cast<int>(std::min(static_cast<float>(width), centerX + sqCovX));
        const int minY = static_cast<int>(std::max(0.0f, centerY - sqCovY));
        const int maxY = static_cast<int>(std::min(static_cast<float>(height), centerY + sqCovY));
        
        const float conic00 = pConics[gaussianIdx * 3 + 0];
        const float conic01 = pConics[gaussianIdx * 3 + 1];
        const float conic11 = pConics[gaussianIdx * 3 + 2];
        
        const float depth = pDepths[gaussianIdx];
        
        // Render to each pixel in bounding box
        for (int y = minY; y < maxY; y++){
            for (int x = minX; x < maxX; x++){
                const int pixIdx = y * width + x;
                
                if (pDone[pixIdx]) continue;
                
                const float dx = x - centerX;
                const float dy = y - centerY; 
                
                const float power = -0.5f * (conic00 * dx * dx + 2.0f * conic01 * dx * dy + conic11 * dy * dy);
                
                if (power > 0.0f) continue;
                
                const float alpha = std::min(0.99f, opacity * std::exp(power));
                if (alpha < 1.0f / 255.0f) continue;
                
                const float testT = pFinalTs[pixIdx] * (1.0f - alpha);
                if (testT < 0.0001f){
                    pDone[pixIdx] = true;
                    continue;
                }
                
                // Record gaussian ID for this pixel
                px2gid[pixIdx].push_back(static_cast<int32_t>(gaussianIdx));
                
                // Accumulate color
                for (int c = 0; c < channels; c++){
                    const float col = pColors[gaussianIdx * channels + c];
                    pOutImg[pixIdx * channels + c] += pFinalTs[pixIdx] * alpha * col;
                }
                
                // Accumulate depth (weighted by alpha)
                pDepthMap[pixIdx] += pFinalTs[pixIdx] * alpha * depth;
                
                // Accumulate alpha
                pAlphaMap[pixIdx] += pFinalTs[pixIdx] * alpha;
                
                pFinalTs[pixIdx] = testT;
            }
        }
    }
    
    // Add background contribution and normalize depth
    for (int pixIdx = 0; pixIdx < width * height; pixIdx++){
        const float finalT = pFinalTs[pixIdx];
        const float totalAlpha = pAlphaMap[pixIdx];
        
        for (int c = 0; c < channels; c++){
            pOutImg[pixIdx * channels + c] += finalT * pBg[c];
        }
        
        // Normalize depth by accumulated alpha to get average depth
        if (totalAlpha > 1e-6f) {
            pDepthMap[pixIdx] /= totalAlpha;
        }
        
        // Reverse the px2gid order (front to back for rendering)
        std::reverse(px2gid[pixIdx].begin(), px2gid[pixIdx].end());
    }
    
    // Save data for backward pass
    ctx->saved_data["xys"] = xys;
    ctx->saved_data["radii"] = radii;
    ctx->saved_data["conics"] = conics;
    ctx->saved_data["colors"] = colors;
    ctx->saved_data["opacity"] = opacity;
    ctx->saved_data["cov2d"] = cov2d;
    ctx->saved_data["camDepths"] = camDepths;
    ctx->saved_data["imgHeight"] = imgHeight;
    ctx->saved_data["imgWidth"] = imgWidth;
    ctx->saved_data["background"] = background;
    ctx->saved_data["finalTs"] = finalTs;
    ctx->saved_data["px2gid"] = reinterpret_cast<int64_t>(px2gid);
    
    return EnhancedRenderOutput{outImg, depthMap, alphaMap, px2gid};
}

tensor_list RasterizeGaussiansCPUEnhanced::backward(AutogradContext *ctx, tensor_list grad_outputs) {
    // Extract saved data
    torch::Tensor xys = ctx->saved_data["xys"].toTensor();
    torch::Tensor radii = ctx->saved_data["radii"].toTensor();
    torch::Tensor conics = ctx->saved_data["conics"].toTensor();
    torch::Tensor colors = ctx->saved_data["colors"].toTensor();
    torch::Tensor opacity = ctx->saved_data["opacity"].toTensor();
    torch::Tensor cov2d = ctx->saved_data["cov2d"].toTensor();
    torch::Tensor camDepths = ctx->saved_data["camDepths"].toTensor();
    int imgHeight = ctx->saved_data["imgHeight"].toInt();
    int imgWidth = ctx->saved_data["imgWidth"].toInt();
    torch::Tensor background = ctx->saved_data["background"].toTensor();
    torch::Tensor finalTs = ctx->saved_data["finalTs"].toTensor();
    const std::vector<int32_t> *px2gid = reinterpret_cast<const std::vector<int32_t> *>(ctx->saved_data["px2gid"].toInt());
    
    // Only handle RGB gradient for now (first element of grad_outputs)
    torch::Tensor grad_out_img = grad_outputs[0];
    
    // Call original backward implementation
    auto result = rasterize_backward_tensor(imgHeight, imgWidth,
                            xys, 
                            radii,
                            conics,
                            colors,
                            opacity,
                            cov2d,
                            camDepths,
                            finalTs,
                            px2gid,
                            grad_out_img
                            );
    
    // Clean up
    delete[] px2gid;
    
    return result;
}

#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)

EnhancedRenderOutput RasterizeGaussiansEnhanced::forward(AutogradContext *ctx, 
        torch::Tensor xys,
        torch::Tensor depths,
        torch::Tensor radii,
        torch::Tensor conics,
        torch::Tensor numTilesHit,
        torch::Tensor colors,
        torch::Tensor opacity,
        int imgHeight,
        int imgWidth,
        torch::Tensor background) {
    
    // Use existing GPU rasterization for RGB
    torch::Tensor rgb = RasterizeGaussians::apply(xys, depths, radii, conics, numTilesHit,
                                                 colors, opacity, imgHeight, imgWidth, background);
    
    // For now, create placeholder depth and alpha maps
    // TODO: Implement GPU-based depth and px2gid extraction
    torch::Device device = xys.device();
    torch::Tensor depthMap = torch::zeros({imgHeight, imgWidth}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor alphaMap = torch::zeros({imgHeight, imgWidth}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Create empty px2gid mapping
    std::vector<int32_t> *px2gid = new std::vector<int32_t>[imgHeight * imgWidth];
    
    // Save context for backward pass
    ctx->saved_data["xys"] = xys;
    ctx->saved_data["depths"] = depths;
    ctx->saved_data["radii"] = radii;
    ctx->saved_data["conics"] = conics;
    ctx->saved_data["numTilesHit"] = numTilesHit;
    ctx->saved_data["colors"] = colors;
    ctx->saved_data["opacity"] = opacity;
    ctx->saved_data["imgHeight"] = imgHeight;
    ctx->saved_data["imgWidth"] = imgWidth;
    ctx->saved_data["background"] = background;
    ctx->saved_data["px2gid"] = reinterpret_cast<int64_t>(px2gid);
    
    return EnhancedRenderOutput{rgb, depthMap, alphaMap, px2gid};
}

tensor_list RasterizeGaussiansEnhanced::backward(AutogradContext *ctx, tensor_list grad_outputs) {
    // Extract saved data
    torch::Tensor xys = ctx->saved_data["xys"].toTensor();
    torch::Tensor depths = ctx->saved_data["depths"].toTensor(); 
    torch::Tensor radii = ctx->saved_data["radii"].toTensor();
    torch::Tensor conics = ctx->saved_data["conics"].toTensor();
    torch::Tensor numTilesHit = ctx->saved_data["numTilesHit"].toTensor();
    torch::Tensor colors = ctx->saved_data["colors"].toTensor();
    torch::Tensor opacity = ctx->saved_data["opacity"].toTensor();
    int imgHeight = ctx->saved_data["imgHeight"].toInt();
    int imgWidth = ctx->saved_data["imgWidth"].toInt();
    torch::Tensor background = ctx->saved_data["background"].toTensor();
    const std::vector<int32_t> *px2gid = reinterpret_cast<const std::vector<int32_t> *>(ctx->saved_data["px2gid"].toInt());
    
    // Use original GPU backward pass for RGB gradient
    torch::Tensor grad_out_img = grad_outputs[0];
    auto result = RasterizeGaussians::backward(ctx, {grad_out_img});
    
    // Clean up
    delete[] px2gid;
    
    return result;
}

#endif

// Utility function for enhanced rendering
EnhancedRenderOutput render_gaussians_enhanced(
    torch::Tensor means,
    torch::Tensor scales,
    torch::Tensor quats,
    torch::Tensor features_dc,
    torch::Tensor features_rest,
    torch::Tensor opacities,
    torch::Tensor viewMat,
    torch::Tensor projMat,
    float fx, float fy, float cx, float cy,
    int height, int width,
    int sh_degree,
    torch::Tensor background,
    const torch::Device& device) {
    
    // Combine spherical harmonics features
    torch::Tensor colors = torch::cat({features_dc.index({torch::indexing::Slice(), 
                                                          torch::indexing::None, 
                                                          torch::indexing::Slice()}), 
                                      features_rest}, 1);
    
    // Project gaussians
    torch::Tensor xys, depths, radii, conics, numTilesHit;
    torch::Tensor cov2d, camDepths;
    
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
#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
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
#endif
    }
    
    // Early exit if no gaussians visible
    if (radii.sum().item<float>() == 0.0f) {
        torch::Tensor rgb = background.repeat({height, width, 1});
        torch::Tensor depth = torch::zeros({height, width}, device);
        torch::Tensor alpha = torch::zeros({height, width}, device);
        std::vector<int32_t>* px2gid = new std::vector<int32_t>[height * width];
        return EnhancedRenderOutput{rgb, depth, alpha, px2gid};
    }
    
    // Compute spherical harmonics colors
    torch::Tensor viewDirs = means.detach() - viewMat.index({torch::indexing::Slice(torch::indexing::None, 3), 
                                                            torch::indexing::Slice(3, 4)}).transpose(0, 1).to(device);
    viewDirs = viewDirs / viewDirs.norm(2, {-1}, true);
    torch::Tensor rgbs;
    
    if (device.is_cpu()) {
        rgbs = SphericalHarmonicsCPU::apply(sh_degree, viewDirs, colors);
    } else {
#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        rgbs = SphericalHarmonics::apply(sh_degree, viewDirs, colors);
#endif
    }
    
    rgbs = torch::clamp_min(rgbs + 0.5f, 0.0f);
    
    // Rasterize with enhanced output
    if (device.is_cpu()) {
        return RasterizeGaussiansCPUEnhanced::apply(xys, radii, conics, rgbs,
                                                   torch::sigmoid(opacities),
                                                   cov2d, camDepths,
                                                   height, width, background);
    } else {
#if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
        return RasterizeGaussiansEnhanced::apply(xys, depths, radii, conics, numTilesHit,
                                               rgbs, torch::sigmoid(opacities),
                                               height, width, background);
#endif
    }
    
    throw std::runtime_error("Device not supported");
}