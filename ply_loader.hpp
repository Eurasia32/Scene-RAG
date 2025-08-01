#ifndef PLY_LOADER_HPP
#define PLY_LOADER_HPP

#include <torch/torch.h>
#include <string>

// Structure to hold gaussian parameters loaded from PLY file
struct PointsTensor {
    torch::Tensor xyz;          // [N, 3] - gaussian centers
    torch::Tensor features_dc;  // [N, 3] - SH DC coefficients (RGB)
    torch::Tensor features_rest; // [N, (max_sh_degree+1)^2-1, 3] - SH higher order coefficients
    torch::Tensor opacities;    // [N, 1] - gaussian opacities
    torch::Tensor scales;       // [N, 3] - gaussian scales
    torch::Tensor quats;        // [N, 4] - gaussian rotations (quaternions)
    
    PointsTensor() = default;
    PointsTensor(torch::Tensor xyz_, torch::Tensor features_dc_, torch::Tensor features_rest_,
                torch::Tensor opacities_, torch::Tensor scales_, torch::Tensor quats_)
        : xyz(xyz_), features_dc(features_dc_), features_rest(features_rest_),
          opacities(opacities_), scales(scales_), quats(quats_) {}
};

// Load gaussian parameters from PLY file
PointsTensor loadPly(const std::string& ply_path);

#endif // PLY_LOADER_HPP