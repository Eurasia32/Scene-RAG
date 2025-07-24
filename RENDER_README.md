# 3D高斯溅射渲染器

这是一个专门用于渲染的3D高斯溅射（3DGS）实现，从原始的opensplat项目中简化而来，专注于渲染功能。

## 功能特点

- **纯渲染功能**: 移除了训练相关代码，专注于高效渲染
- **简化的输入**: 只需要PLY文件、降采样倍数、球谐函数系数和视角变换矩阵
- **灵活的视角控制**: 支持任意相机视角的渲染
- **降采样支持**: 可以指定降采样倍数来控制输出分辨率
- **CPU/GPU支持**: 自动检测并使用可用的CUDA设备

## 输入参数

### 必需参数
- `-i, --input`: 输入的PLY模型文件路径
- `-m, --view-matrix`: 16个元素的视图变换矩阵（行主序，camera-to-world）

### 可选参数
- `-o, --output`: 输出图像路径（默认: output.png）
- `-d, --downscale`: 降采样倍数（默认: 1.0）
- `-s, --sh-degree`: 球谐函数阶数（默认: 3）
- `--width`: 图像宽度（默认: 800）
- `--height`: 图像高度（默认: 600）
- `--fx`: 焦距fx（默认: 400.0）
- `--fy`: 焦距fy（默认: 400.0）

## 构建方法

### 前提条件
- CMake >= 3.21
- C++17编译器
- PyTorch C++ API (libtorch)
- OpenCV
- CUDA（可选，用于GPU加速）

### 构建步骤

1. 使用专用的CMakeLists文件：
```bash
# 复制专用的CMakeLists文件
cp CMakeLists_render.txt CMakeLists.txt

# 创建构建目录
mkdir build && cd build

# 配置和构建
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/opencv"
make -j8
```

2. 生成的可执行文件为 `opensplat_render`

## 使用示例

### 基本用法
```bash
./opensplat_render \
  -i model.ply \
  -o rendered_image.png \
  -m 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0
```

### 带降采样的渲染
```bash
./opensplat_render \
  -i model.ply \
  -o low_res.png \
  -d 2.0 \
  --width 1600 \
  --height 1200 \
  -m "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"
```

### 自定义球谐函数阶数
```bash
./opensplat_render \
  -i model.ply \
  -s 2 \
  -m "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"
```

## 视图矩阵格式

视图矩阵是一个4x4的camera-to-world变换矩阵，以行主序传递：

```
[ R11 R12 R13 Tx ]
[ R21 R22 R23 Ty ]  → "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"
[ R31 R32 R33 Tz ]
[  0   0   0   1 ]
```

其中：
- R: 3x3旋转矩阵
- T: 3x1平移向量
- 上述示例表示相机位于(0,0,5)位置，朝向-Z方向

## 程序输出

程序会输出：
- 加载的高斯点数量
- 使用的计算设备（CPU/GPU）
- 渲染进度信息
- 最终的渲染尺寸
- 保存路径

## 性能优化

- 使用GPU可显著提升渲染速度
- 降采样可以减少计算量，适用于预览
- 较低的球谐函数阶数可以加快渲染但可能影响质量

## 输出格式

输出图像为PNG格式，支持RGB颜色。图像会自动进行颜色空间转换以确保正确显示。