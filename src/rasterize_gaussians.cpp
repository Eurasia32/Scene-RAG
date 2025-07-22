#include "rasterize_gaussians.hpp"
#include <gsplat-cpu/bindings.h>

torch::Tensor RasterizeGaussiansCPU::forward(AutogradContext *ctx, 
            torch::Tensor xys,
            torch::Tensor radii,
            torch::Tensor conics,
            torch::Tensor colors,
            torch::Tensor opacity,
            torch::Tensor cov2d,
            torch::Tensor camDepths,
            int imgHeight,
            int imgWidth,
            torch::Tensor background
        ){
    
    int numPoints = xys.size(0);

    auto t = rasterize_forward_tensor_cpu(imgWidth, imgHeight, 
                            xys,
                            conics,
                            colors,
                            opacity,
                            background,
                            cov2d,
                            camDepths
                            );
    // Final image
    torch::Tensor outImg = std::get<0>(t);

    torch::Tensor finalTs = std::get<1>(t);
    std::vector<int32_t> *px2gid = std::get<2>(t);

    ctx->saved_data["px2gid"] = reinterpret_cast<int64_t>(px2gid);
    ctx->saved_data["imgWidth"] = imgWidth;
    ctx->saved_data["imgHeight"] = imgHeight;
    ctx->save_for_backward({ xys, conics, colors, opacity, background, cov2d, camDepths, finalTs });
    
    return outImg;
}

tensor_list RasterizeGaussiansCPU::backward(AutogradContext *ctx, tensor_list grad_outputs) {
    torch::Tensor v_outImg = grad_outputs[0];
    int imgHeight = ctx->saved_data["imgHeight"].toInt();
    int imgWidth = ctx->saved_data["imgWidth"].toInt();
    const std::vector<int32_t> *px2gid = reinterpret_cast<const std::vector<int32_t> *>(ctx->saved_data["px2gid"].toInt());

    variable_list saved = ctx->get_saved_variables();
    torch::Tensor xys = saved[0];
    torch::Tensor conics = saved[1];
    torch::Tensor colors = saved[2];
    torch::Tensor opacity = saved[3];
    torch::Tensor background = saved[4];
    torch::Tensor cov2d = saved[5];
    torch::Tensor camDepths = saved[6];
    torch::Tensor finalTs = saved[7];

    torch::Tensor v_outAlpha = torch::zeros_like(v_outImg.index({"...", 0}));
    
    auto t = rasterize_backward_tensor_cpu(imgHeight, imgWidth, 
                            xys,
                            conics,
                            colors,
                            opacity,
                            background,
                            cov2d,
                            camDepths,
                            finalTs,
                            px2gid,
                            v_outImg,
                            v_outAlpha);

    delete[] px2gid;


    torch::Tensor v_xy = std::get<0>(t);
    torch::Tensor v_conic = std::get<1>(t);
    torch::Tensor v_colors = std::get<2>(t);
    torch::Tensor v_opacity = std::get<3>(t);
    torch::Tensor none;

    return { v_xy,
            none, // radii
            v_conic,
            v_colors,
            v_opacity,
            none, // cov2d
            none, // camDepths
            none, // imgHeight
            none, // imgWidth
            none // background
    };
}


