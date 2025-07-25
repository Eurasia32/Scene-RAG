# GSRender Python接口编译和使用指南

## 系统要求

### 基础依赖
- **C++17兼容编译器** (GCC 7+, Clang 6+, MSVC 2019+)
- **CMake 3.21+**
- **Python 3.7+**
- **PyTorch 1.9+** (包含C++扩展)
- **OpenCV 4.5+**
- **Eigen3**

### Python依赖
```bash
pip install torch torchvision numpy opencv-python pybind11
```

### 可选依赖 (用于CLIP和SAM集成)
```bash
# CLIP
pip install git+https://github.com/openai/CLIP.git pillow

# SAM  
pip install git+https://github.com/facebookresearch/segment-anything.git

# 下载SAM模型权重
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## 编译步骤

### 方法1: 使用setup.py (推荐)

```bash
# 1. 克隆或进入项目目录
cd Scene-RAG

# 2. 编译Python扩展 (开发模式)
python setup.py build_ext --inplace

# 3. 或安装到Python环境
pip install -e .

# 4. 测试安装
python -c "import gsrender; print('✓ GSRender导入成功')"
```

### 方法2: 使用CMake + make

```bash
# 1. 创建构建目录
mkdir -p build && cd build

# 2. 配置CMake (指定libtorch路径)
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch" -DWITH_PYTHON=ON

# 3. 编译
make -j$(nproc)

# 4. 安装Python模块
make install_python
```

### 方法3: 使用PyTorch的cpp_extension

```bash
# 创建临时编译脚本
cat > compile_gsrender.py << 'EOF'
from torch.utils.cpp_extension import load
import os

# 源文件列表
sources = [
    "src/python_bindings.cpp",
    "src/gsrender_interface.cpp", 
    "src/model_render.cpp",
    "src/cv_utils_render.cpp",
    "src/rasterizer/gsplat_render.cpp"
]

# 包含目录
include_dirs = ["include", "src/rasterizer"]

# 编译并加载模块
gsrender = load(
    name="gsrender",
    sources=sources,
    extra_include_paths=include_dirs,
    extra_cflags=["-std=c++17", "-O3"],
    verbose=True
)

print("✓ GSRender编译并加载成功")
EOF

python compile_gsrender.py
```

## 环境配置

### CUDA支持 (可选)
如果系统支持CUDA，确保安装了对应版本的PyTorch:
```bash
# 检查CUDA版本
nvidia-smi

# 安装CUDA版本的PyTorch (以CUDA 11.8为例)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### OpenCV配置
如果系统中有多个OpenCV版本，可以指定路径:
```bash
export OpenCV_DIR="/path/to/opencv"
cmake .. -DOpenCV_DIR=$OpenCV_DIR
```

### libtorch配置
下载并配置libtorch:
```bash
# 下载libtorch (CPU版本)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-*.zip

# 或CUDA版本
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip

# 设置环境变量
export CMAKE_PREFIX_PATH="/path/to/libtorch:$CMAKE_PREFIX_PATH"
```

## 使用示例

### 基础渲染
```python
import gsrender
import numpy as np

# 创建渲染器
renderer = gsrender.GSRenderInterface() 

# 加载模型
success = renderer.load_model("model.ply", "cuda")
if not success:
    print("模型加载失败")
    exit(1)

# 创建相机参数
view_matrix = np.eye(4, dtype=np.float32)
view_matrix[2, 3] = 5.0  # 相机Z位置

camera_params = gsrender.create_camera_params(
    view_matrix=view_matrix,
    width=800, height=600,
    fx=400.0, fy=400.0
)

# 渲染
result = renderer.render(camera_params)

print(f"渲染完成: {result.width}x{result.height}")
print(f"可见高斯点: {result.visible_gaussians}")
print(f"渲染时间: {result.render_time_ms:.2f}ms")

# 获取结果
rgb_image = result.rgb_image  # numpy数组 [H, W, 3]
depth_image = result.depth_image  # numpy数组 [H, W, 1]
px2gid_mapping = result.pixel_to_gaussian_mapping  # 像素到高斯点映射
```

### 高斯点过滤
```python
# 获取模型信息
model_info = renderer.get_model_info()
print(f"总高斯点数: {model_info.num_gaussians}")

# 按不透明度过滤
high_opacity_indices = renderer.filter_gaussians(
    filter_by_opacity=True,
    min_opacity=0.5
)

print(f"高不透明度高斯点: {len(high_opacity_indices)}")

# 使用过滤后的高斯点渲染
filtered_result = renderer.render(camera_params, high_opacity_indices)
```

### 批量渲染
```python
# 创建多个视角
camera_params_list = []
for i in range(5):
    angle = i * 2 * np.pi / 5
    view_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle), 5*np.cos(angle)],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 5*np.sin(angle)],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    params = gsrender.create_camera_params(view_matrix, 640, 480, 320, 320)
    camera_params_list.append(params)

# 批量渲染
results = renderer.render_batch(camera_params_list)
print(f"批量渲染完成: {len(results)} 个视角")
```

## 高级集成

### 与CLIP集成
```python
import clip
from PIL import Image

# 加载CLIP模型
clip_model, clip_preprocess = clip.load("ViT-B/32")

# 渲染场景
result = renderer.render(camera_params)

# 提取CLIP特征
pil_image = Image.fromarray((result.rgb_image * 255).astype(np.uint8))
image_input = clip_preprocess(pil_image).unsqueeze(0)

with torch.no_grad():
    image_features = clip_model.encode_image(image_input)

# 文本查询
text_queries = ["a chair", "a table", "a room"]
text_tokens = clip.tokenize(text_queries)

with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)

# 计算相似度
similarities = (text_features @ image_features.T).softmax(dim=0)
for i, query in enumerate(text_queries):
    print(f"{query}: {similarities[i].item():.3f}")
```

### 与SAM集成
```python
from segment_anything import sam_model_registry, SamPredictor

# 加载SAM模型
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# 渲染场景
result = renderer.render(camera_params)
image_uint8 = (result.rgb_image * 255).astype(np.uint8)

# 设置图像
predictor.set_image(image_uint8)

# 点击分割
input_point = np.array([[400, 300]])  # 图像中心
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# 分析分割区域对应的高斯点
for i, mask in enumerate(masks):
    gaussian_ids = set()
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:
                pixel_gaussians = result.get_pixel_gaussians(x, y)
                gaussian_ids.update(pixel_gaussians)
    
    print(f"分割区域 {i}: {len(gaussian_ids)} 个高斯点")
```

## 性能优化建议

1. **GPU内存管理**:
   ```python
   # 定期清理GPU内存
   renderer.clear_memory()
   
   # 使用较小的图像尺寸进行开发
   camera_params = gsrender.create_camera_params(
       view_matrix, 320, 240, 160, 160  # 小尺寸
   )
   ```

2. **批量处理**:
   ```python
   # 批量渲染比循环渲染更高效
   results = renderer.render_batch(camera_params_list)
   ```

3. **高斯点过滤**:
   ```python
   # 预先过滤不需要的高斯点
   visible_indices = renderer.filter_gaussians(
       filter_by_opacity=True, min_opacity=0.1
   )
   ```

4. **预热**:
   ```python
   # 首次渲染前预热GPU
   renderer.warmup(camera_params)
   ```

## 故障排除

### 编译错误

1. **找不到PyTorch**:
   ```bash
   # 确保PyTorch正确安装
   python -c "import torch; print(torch.__version__)"
   
   # 如果使用conda，可能需要设置
   export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
   ```

2. **找不到OpenCV**:
   ```bash
   # 安装OpenCV开发包
   sudo apt-get install libopencv-dev  # Ubuntu
   brew install opencv                 # macOS
   
   # 或指定OpenCV路径
   cmake .. -DOpenCV_DIR="/path/to/opencv"
   ```

3. **CUDA链接错误**:
   ```bash
   # 确保CUDA版本匹配
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

### 运行时错误

1. **模型加载失败**:
   - 检查PLY文件格式是否正确
   - 确保文件路径存在
   - 检查文件权限

2. **GPU内存不足**:
   ```python
   # 减少图像分辨率
   # 使用CPU模式
   renderer.load_model(ply_path, "cpu")
   ```

3. **渲染结果为黑色**:
   - 检查相机位姿是否正确
   - 确保场景在相机视野内
   - 调整相机参数 (fx, fy, near_plane, far_plane)

## 完整示例

查看项目的 `examples/` 目录获取完整示例:
- `basic_rendering.py`: 基础渲染功能
- `clip_sam_integration.py`: CLIP和SAM集成示例

运行示例:
```bash
cd examples
python basic_rendering.py
python clip_sam_integration.py
```

## API参考

详细的API文档请参考源代码中的注释和类型提示。主要类和函数:

- `GSRenderInterface`: 主渲染接口
- `CameraParams`: 相机参数结构
- `RenderResult`: 渲染结果结构
- `create_camera_params()`: 创建相机参数的工具函数
- `tensor_to_numpy()`: tensor转换工具函数