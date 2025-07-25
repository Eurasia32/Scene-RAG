#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "gsrender_interface.hpp"

namespace py = pybind11;

// 辅助函数：将numpy数组转换为torch tensor
torch::Tensor numpy_to_tensor(py::array_t<float> input) {
    py::buffer_info buf_info = input.request();
    
    std::vector<int64_t> shape;
    for (auto s : buf_info.shape) {
        shape.push_back(s);
    }
    
    torch::Tensor tensor = torch::from_blob(
        buf_info.ptr, 
        shape, 
        torch::kFloat32
    ).clone();
    
    return tensor;
}

// 辅助函数：将torch tensor转换为numpy数组
py::array_t<float> tensor_to_numpy(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.cpu().contiguous();
    
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
    
    for (int i = 0; i < cpu_tensor.dim(); i++) {
        shape.push_back(cpu_tensor.size(i));
        strides.push_back(cpu_tensor.stride(i) * sizeof(float));
    }
    
    return py::array_t<float>(
        shape,
        strides,
        cpu_tensor.data_ptr<float>(),
        py::cast(cpu_tensor) // 保持tensor存活
    );
}

// pybind11模块定义
PYBIND11_MODULE(gsrender, m) {
    m.doc() = "GSRender: 3D Gaussian Splatting渲染接口，集成CLIP和SAM支持";
    
    // CameraParams结构体绑定
    py::class_<CameraParams>(m, "CameraParams")
        .def(py::init<>())
        .def_readwrite("fx", &CameraParams::fx)
        .def_readwrite("fy", &CameraParams::fy)
        .def_readwrite("cx", &CameraParams::cx)
        .def_readwrite("cy", &CameraParams::cy)
        .def_readwrite("width", &CameraParams::width)
        .def_readwrite("height", &CameraParams::height)
        .def_readwrite("downscale_factor", &CameraParams::downscale_factor)
        .def_readwrite("sh_degree", &CameraParams::sh_degree)
        .def_readwrite("near_plane", &CameraParams::near_plane)
        .def_readwrite("far_plane", &CameraParams::far_plane)
        .def_property("view_matrix",
            [](const CameraParams& self) {
                return tensor_to_numpy(self.view_matrix);
            },
            [](CameraParams& self, py::array_t<float> input) {
                self.view_matrix = numpy_to_tensor(input);
            })
        .def("set_view_matrix", [](CameraParams& self, py::array_t<float> matrix) {
            self.view_matrix = numpy_to_tensor(matrix);
        }, "设置视图矩阵 (4x4 numpy数组)")
        .def("__repr__", [](const CameraParams& self) {
            return "<CameraParams: " + std::to_string(self.width) + "x" + std::to_string(self.height) + ">";
        });
    
    // RenderResult结构体绑定
    py::class_<RenderResult>(m, "RenderResult")
        .def(py::init<>())
        .def_property_readonly("rgb_image", [](const RenderResult& self) {
            return tensor_to_numpy(self.rgb_image);
        })
        .def_property_readonly("depth_image", [](const RenderResult& self) {
            return tensor_to_numpy(self.depth_image);
        })
        .def_property_readonly("final_transmittance", [](const RenderResult& self) {
            return tensor_to_numpy(self.final_transmittance);
        })
        .def_readwrite("pixel_to_gaussian_mapping", &RenderResult::pixel_to_gaussian_mapping)
        .def_readwrite("width", &RenderResult::width)
        .def_readwrite("height", &RenderResult::height)
        .def_readwrite("visible_gaussians", &RenderResult::visible_gaussians)
        .def_readwrite("render_time_ms", &RenderResult::render_time_ms)
        .def("get_pixel_gaussians", [](const RenderResult& self, int x, int y) {
            if (x < 0 || x >= self.width || y < 0 || y >= self.height) {
                throw std::out_of_range("像素坐标超出范围");
            }
            size_t pixIdx = y * self.width + x;
            return self.pixel_to_gaussian_mapping[pixIdx];
        }, "获取指定像素位置的高斯点ID列表", py::arg("x"), py::arg("y"))
        .def("get_px2gid_tensor", [](const RenderResult& self, int max_gaussians_per_pixel = 10) {
            torch::Tensor result = pixelMappingToTensor(
                self.pixel_to_gaussian_mapping, 
                self.width, 
                self.height, 
                max_gaussians_per_pixel
            );
            return tensor_to_numpy(result);
        }, "将px2gid映射转换为密集tensor格式", py::arg("max_gaussians_per_pixel") = 10)
        .def("__repr__", [](const RenderResult& self) {
            return "<RenderResult: " + std::to_string(self.width) + "x" + std::to_string(self.height) + 
                   ", " + std::to_string(self.visible_gaussians) + " gaussians, " + 
                   std::to_string(self.render_time_ms) + "ms>";
        });
    
    // ModelInfo结构体绑定
    py::class_<GSRenderInterface::ModelInfo>(m, "ModelInfo")
        .def(py::init<>())
        .def_readwrite("num_gaussians", &GSRenderInterface::ModelInfo::num_gaussians)
        .def_readwrite("sh_degree", &GSRenderInterface::ModelInfo::sh_degree)
        .def_property_readonly("means", [](const GSRenderInterface::ModelInfo& self) {
            return tensor_to_numpy(self.means);
        })
        .def_property_readonly("scales", [](const GSRenderInterface::ModelInfo& self) {
            return tensor_to_numpy(self.scales);
        })
        .def_property_readonly("rotations", [](const GSRenderInterface::ModelInfo& self) {
            return tensor_to_numpy(self.rotations);
        })
        .def_property_readonly("opacities", [](const GSRenderInterface::ModelInfo& self) {
            return tensor_to_numpy(self.opacities);
        })
        .def_property_readonly("colors_dc", [](const GSRenderInterface::ModelInfo& self) {
            return tensor_to_numpy(self.colors_dc);
        })
        .def_property_readonly("colors_rest", [](const GSRenderInterface::ModelInfo& self) {
            return tensor_to_numpy(self.colors_rest);
        })
        .def("__repr__", [](const GSRenderInterface::ModelInfo& self) {
            return "<ModelInfo: " + std::to_string(self.num_gaussians) + " gaussians>";
        });
    
    // GSRenderInterface主类绑定
    py::class_<GSRenderInterface>(m, "GSRenderInterface")
        .def(py::init<>())
        .def("load_model", &GSRenderInterface::loadModel,
             "加载3DGS模型", py::arg("ply_path"), py::arg("device") = "cpu")
        .def("render", &GSRenderInterface::render,
             "渲染场景", py::arg("camera_params"), py::arg("gaussian_indices") = std::nullopt)
        .def("render_batch", &GSRenderInterface::renderBatch,
             "批量渲染多个视角", py::arg("camera_params_list"), py::arg("gaussian_indices") = std::nullopt)
        .def("get_model_info", &GSRenderInterface::getModelInfo,
             "获取模型信息")
        .def("filter_gaussians", &GSRenderInterface::filterGaussians,
             "根据条件过滤高斯点",
             py::arg("filter_by_position") = false,
             py::arg("position_bounds") = std::vector<float>(),
             py::arg("filter_by_opacity") = false,
             py::arg("min_opacity") = 0.1f)
        .def("set_background_color", &GSRenderInterface::setBackgroundColor,
             "设置背景颜色", py::arg("r"), py::arg("g"), py::arg("b"))
        .def("get_device", &GSRenderInterface::getDevice,
             "获取计算设备")
        .def("clear_memory", &GSRenderInterface::clearMemory,
             "释放GPU内存")
        .def("warmup", &GSRenderInterface::warmup,
             "预热GPU", py::arg("camera_params"));
    
    // 工具函数绑定
    m.def("create_camera_params", [](py::array_t<float> view_matrix,
                                    int width, int height,
                                    float fx, float fy,
                                    float cx = -1, float cy = -1,
                                    float downscale_factor = 1.0f,
                                    int sh_degree = 3) {
        // 将numpy数组转换为vector
        py::buffer_info buf = view_matrix.request();
        if (buf.size != 16) {
            throw std::invalid_argument("视图矩阵必须是4x4矩阵(16个元素)");
        }
        
        std::vector<float> matrix_flat;
        float* ptr = static_cast<float*>(buf.ptr);
        for (int i = 0; i < 16; i++) {
            matrix_flat.push_back(ptr[i]);
        }
        
        return createCameraParams(matrix_flat, width, height, fx, fy, cx, cy, downscale_factor, sh_degree);
    }, "从numpy数组创建相机参数",
    py::arg("view_matrix"), py::arg("width"), py::arg("height"),
    py::arg("fx"), py::arg("fy"), py::arg("cx") = -1, py::arg("cy") = -1,
    py::arg("downscale_factor") = 1.0f, py::arg("sh_degree") = 3);
    
    m.def("tensor_to_vector", [](py::array_t<float> input) {
        torch::Tensor tensor = numpy_to_tensor(input);
        return tensorToVector(tensor);
    }, "将tensor转换为vector");
    
    m.def("pixel_mapping_to_tensor", [](const std::vector<std::vector<int32_t>>& mapping,
                                       int width, int height, int max_gaussians_per_pixel = 10) {
        torch::Tensor result = pixelMappingToTensor(mapping, width, height, max_gaussians_per_pixel);
        return tensor_to_numpy(result);
    }, "将像素-高斯映射转换为密集tensor",
    py::arg("mapping"), py::arg("width"), py::arg("height"), py::arg("max_gaussians_per_pixel") = 10);
    
    // 版本信息
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "GSRender Team";
    m.attr("__description__") = "3D Gaussian Splatting渲染接口，支持CLIP和SAM集成";
}