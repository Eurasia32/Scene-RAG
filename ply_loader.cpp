#include "ply_loader.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iostream>

PointsTensor loadPly(const std::string& ply_path) {
    std::ifstream file(ply_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open PLY file: " + ply_path);
    }
    
    std::string line;
    bool header_ended = false;
    int vertex_count = 0;
    bool binary_format = false;
    
    // Parse header
    while (std::getline(file, line) && !header_ended) {
        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string element, vertex;
            iss >> element >> vertex >> vertex_count;
        } else if (line.find("format binary") != std::string::npos) {
            binary_format = true;
        } else if (line == "end_header") {
            header_ended = true;
        }
    }
    
    if (vertex_count == 0) {
        throw std::runtime_error("No vertices found in PLY file");
    }
    
    // Allocate tensors
    auto xyz = torch::zeros({vertex_count, 3}, torch::kFloat32);
    auto features_dc = torch::zeros({vertex_count, 3}, torch::kFloat32);
    auto features_rest = torch::zeros({vertex_count, 15, 3}, torch::kFloat32); // 15 coefficients for higher order (degree 1-3)
    auto opacities = torch::zeros({vertex_count, 1}, torch::kFloat32);
    auto scales = torch::zeros({vertex_count, 3}, torch::kFloat32);
    auto quats = torch::zeros({vertex_count, 4}, torch::kFloat32);
    
    auto xyz_ptr = xyz.accessor<float, 2>();
    auto features_dc_ptr = features_dc.accessor<float, 2>();
    auto features_rest_ptr = features_rest.accessor<float, 3>();
    auto opacities_ptr = opacities.accessor<float, 2>();
    auto scales_ptr = scales.accessor<float, 2>();
    auto quats_ptr = quats.accessor<float, 2>();
    
    if (binary_format) {
        // Read binary data
        for (int i = 0; i < vertex_count; i++) {
            // Read position (x, y, z)
            file.read(reinterpret_cast<char*>(&xyz_ptr[i][0]), sizeof(float));
            file.read(reinterpret_cast<char*>(&xyz_ptr[i][1]), sizeof(float));
            file.read(reinterpret_cast<char*>(&xyz_ptr[i][2]), sizeof(float));
            
            // Read normals (skip - nx, ny, nz)
            float temp;
            file.read(reinterpret_cast<char*>(&temp), sizeof(float));
            file.read(reinterpret_cast<char*>(&temp), sizeof(float));
            file.read(reinterpret_cast<char*>(&temp), sizeof(float));
            
            // Read SH features - first 3 are DC (f_dc_0, f_dc_1, f_dc_2)
            file.read(reinterpret_cast<char*>(&features_dc_ptr[i][0]), sizeof(float));
            file.read(reinterpret_cast<char*>(&features_dc_ptr[i][1]), sizeof(float));
            file.read(reinterpret_cast<char*>(&features_dc_ptr[i][2]), sizeof(float));
            
            // Read remaining SH features (15 coefficients for degree 1-3)
            for (int j = 0; j < 15; j++) {
                for (int k = 0; k < 3; k++) {
                    file.read(reinterpret_cast<char*>(&features_rest_ptr[i][j][k]), sizeof(float));
                }
            }
            
            // Skip remaining SH coefficients if present (45-15=30 coefficients)
            for (int j = 0; j < 30; j++) {
                for (int k = 0; k < 3; k++) {
                    file.read(reinterpret_cast<char*>(&temp), sizeof(float));
                }
            }
            
            // Read opacity
            file.read(reinterpret_cast<char*>(&opacities_ptr[i][0]), sizeof(float));
            
            // Read scales (scale_0, scale_1, scale_2)
            file.read(reinterpret_cast<char*>(&scales_ptr[i][0]), sizeof(float));
            file.read(reinterpret_cast<char*>(&scales_ptr[i][1]), sizeof(float));
            file.read(reinterpret_cast<char*>(&scales_ptr[i][2]), sizeof(float));
            
            // Read quaternion (rot_0, rot_1, rot_2, rot_3)
            file.read(reinterpret_cast<char*>(&quats_ptr[i][0]), sizeof(float));
            file.read(reinterpret_cast<char*>(&quats_ptr[i][1]), sizeof(float));
            file.read(reinterpret_cast<char*>(&quats_ptr[i][2]), sizeof(float));
            file.read(reinterpret_cast<char*>(&quats_ptr[i][3]), sizeof(float));
        }
    } else {
        // Read ASCII data (simplified - assumes standard OpenSplat PLY format)
        for (int i = 0; i < vertex_count; i++) {
            if (!std::getline(file, line)) {
                throw std::runtime_error("Unexpected end of file while reading vertices");
            }
            
            std::istringstream iss(line);
            float x, y, z, nx, ny, nz;
            iss >> x >> y >> z >> nx >> ny >> nz;
            
            xyz_ptr[i][0] = x;
            xyz_ptr[i][1] = y;
            xyz_ptr[i][2] = z;
            
            // Read SH DC coefficients
            iss >> features_dc_ptr[i][0] >> features_dc_ptr[i][1] >> features_dc_ptr[i][2];
            
            // Read remaining SH coefficients (only first 15 for degree 1-3)
            for (int j = 0; j < 15; j++) {
                for (int k = 0; k < 3; k++) {
                    iss >> features_rest_ptr[i][j][k];
                }
            }
            
            // Skip remaining SH coefficients if present (30 more)
            float temp;
            for (int j = 0; j < 30; j++) {
                for (int k = 0; k < 3; k++) {
                    iss >> temp; // Just read and discard
                }
            }
            
            // Read opacity
            iss >> opacities_ptr[i][0];
            
            // Read scales
            iss >> scales_ptr[i][0] >> scales_ptr[i][1] >> scales_ptr[i][2];
            
            // Read quaternion
            iss >> quats_ptr[i][0] >> quats_ptr[i][1] >> quats_ptr[i][2] >> quats_ptr[i][3];
        }
    }
    
    file.close();
    
    return PointsTensor(xyz, features_dc, features_rest, opacities, scales, quats);
}