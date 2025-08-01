#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include "python_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(opensplat_render, m) {
    m.doc() = "OpenSplat Python Rendering Interface";
    
    // Gaussian parameters structure
    py::class_<GaussianParams>(m, "GaussianParams")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, 
                     torch::Tensor, torch::Tensor, torch::Tensor>(),
             "Create GaussianParams",
             py::arg("means"), py::arg("scales"), py::arg("quats"),
             py::arg("features_dc"), py::arg("features_rest"), py::arg("opacities"))
        .def_readwrite("means", &GaussianParams::means, "Gaussian centers [N, 3]")
        .def_readwrite("scales", &GaussianParams::scales, "Gaussian scales [N, 3]")
        .def_readwrite("quats", &GaussianParams::quats, "Gaussian rotations as quaternions [N, 4]")
        .def_readwrite("features_dc", &GaussianParams::features_dc, "DC spherical harmonics [N, 3]")
        .def_readwrite("features_rest", &GaussianParams::features_rest, "Higher order SH [(deg+1)^2-1, 3]")
        .def_readwrite("opacities", &GaussianParams::opacities, "Gaussian opacities [N, 1]");
    
    // Camera parameters structure
    py::class_<CameraParams>(m, "CameraParams")
        .def(py::init<float, float, float, float, int, int, torch::Tensor>(),
             "Create CameraParams",
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
             py::arg("width"), py::arg("height"), py::arg("world_to_cam"))
        .def_readwrite("fx", &CameraParams::fx, "Focal length X")
        .def_readwrite("fy", &CameraParams::fy, "Focal length Y")
        .def_readwrite("cx", &CameraParams::cx, "Principal point X")
        .def_readwrite("cy", &CameraParams::cy, "Principal point Y")
        .def_readwrite("width", &CameraParams::width, "Image width")
        .def_readwrite("height", &CameraParams::height, "Image height")
        .def_readwrite("world_to_cam", &CameraParams::world_to_cam, "World to camera transformation [4, 4]");
    
    // Render output structure  
    py::class_<RenderOutput>(m, "RenderOutput", "Rendering output containing RGB, depth and pixel-to-gaussian mapping")
        .def_readwrite("rgb", &RenderOutput::rgb, "Rendered RGB image [H, W, 3]")
        .def_readwrite("depth", &RenderOutput::depth, "Depth map [H, W]")
        .def_readwrite("px2gid", &RenderOutput::px2gid, "Pixel to gaussian ID mapping [H, W, max_gaussians]");
    
    // Main renderer class
    py::class_<GaussianRenderer>(m, "GaussianRenderer")
        .def(py::init<const std::string&, int>(),
             "Create GaussianRenderer",
             py::arg("device") = "cuda", py::arg("sh_degree") = 3)
        .def("render", &GaussianRenderer::render,
             "Render single view",
             py::arg("gaussians"), py::arg("camera"), 
             py::arg("downsample_factor") = 1.0f,
             py::arg("background") = py::none())
        .def("render_batch", &GaussianRenderer::render_batch,
             "Render multiple views",
             py::arg("gaussians"), py::arg("cameras"),
             py::arg("downsample_factor") = 1.0f,
             py::arg("background") = py::none())
        .def("set_device", &GaussianRenderer::set_device,
             "Set computing device", py::arg("device"))
        .def("get_device", &GaussianRenderer::get_device,
             "Get current computing device")
        .def_static("load_gaussians", &GaussianRenderer::load_gaussians,
                   "Load gaussians from PLY file", py::arg("ply_path"))
        .def_static("create_camera", &GaussianRenderer::create_camera,
                   "Create camera parameters",
                   py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
                   py::arg("width"), py::arg("height"), py::arg("world_to_cam_matrix"));
    
    // Utility functions
    m.def("version", []() {
        return "OpenSplat Python Renderer v1.0.0";
    }, "Get version information");
    
    m.def("available_devices", []() {
        std::vector<std::string> devices = {"cpu"};
        if (torch::cuda::is_available()) {
            devices.push_back("cuda");
        }
        return devices;
    }, "Get list of available devices");
}