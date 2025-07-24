# 可以删除的不必要文件列表

## 对于渲染专用工程，以下文件可以删除：

### 不再需要的源文件：
- `src/opensplat.cpp` (原始训练+渲染程序，已被opensplat_render.cpp替代)
- `src/model.cpp` (已被model_render.cpp替代)
- `src/cv_utils.cpp` (已被cv_utils_render.cpp替代)
- `src/point_io.cpp` (PointSet相关功能不需要)

### 不再需要的头文件：
- `include/model.hpp` (已被model_render.hpp替代)
- `include/cv_utils.hpp` (已被cv_utils_render.hpp替代)
- `include/point_io.hpp` (PointSet相关功能不需要)
- `include/opensplat.hpp` (基本为空，只包含model.hpp)
- `include/cam.hpp` (如果没有用到相机相关的训练功能)

### 可能可以删除的文件（需要检查依赖）：
- `src/gsplat_cpu.cpp` (检查是否与rasterizer重复)
- 其他训练相关的工具文件

### 需要保留的核心文件：
- `src/opensplat_render.cpp` (新的渲染主程序)
- `src/model_render.cpp` (简化的模型加载)
- `src/cv_utils_render.cpp` (简化的图像处理)
- `include/model_render.hpp` (简化的模型头文件)
- `include/cv_utils_render.hpp` (简化的图像处理头文件)
- `include/constants.hpp` (常量定义)
- `src/rasterizer/` 目录下的所有文件 (核心渲染算法)
- `CMakeLists.txt` (已更新为使用简化文件)

## 清理命令：

如果您确认要删除这些文件，可以执行：

```bash
# 删除不需要的源文件
rm src/opensplat.cpp
rm src/model.cpp  
rm src/cv_utils.cpp
rm src/point_io.cpp

# 删除不需要的头文件
rm include/model.hpp
rm include/cv_utils.hpp
rm include/point_io.hpp
rm include/opensplat.hpp

# 如果cam.hpp没有被使用，也可以删除
# rm include/cam.hpp
```

**注意**: 在执行删除操作之前，建议先备份原始文件或使用git提交当前状态。