# Scene-RAG: 3D Gaussian Splatting场景语义检索增强生成系统

## 概述

Scene-RAG是一个集成在3D Gaussian Splatting渲染器中的智能场景理解和检索系统。它能够：

1. **语义分割**: 对渲染结果进行自动语义分割
2. **特征提取**: 使用CLIP模型提取视觉特征
3. **向量存储**: 将特征向量存储到向量数据库中进行聚类
4. **3D映射**: 建立2D像素与3D高斯点的对应关系
5. **智能查询**: 支持自然语言查询和多模态检索

## 系统架构

```
渲染结果 → 语义分割 → CLIP特征提取 → 向量数据库 → RAG查询
    ↓           ↓            ↓            ↓         ↓
RGB图像    分割掩码     特征向量      聚类存储    智能检索
    ↓           ↓            ↓            ↓         ↓
px2gid映射  边界框信息   512维特征    相似度搜索  结果排序
```

## 核心组件

### 1. Pixel2GaussianMapper
- **功能**: 建立2D像素到3D高斯点的映射关系
- **输入**: px2gid数组（渲染器输出）
- **输出**: 像素-高斯点权重映射

### 2. SegmentationModule
- **功能**: 对渲染图像进行语义分割
- **当前实现**: K-means颜色聚类（简化版）
- **未来**: 集成SAM (Segment Anything Model)
- **输出**: 分割掩码列表

### 3. CLIPFeatureExtractor
- **功能**: 提取图像区域的CLIP特征向量
- **当前实现**: HSV颜色直方图（简化版）
- **未来**: 真实CLIP模型集成
- **输出**: 512维特征向量

### 4. VectorDatabase
- **功能**: 存储和检索特征向量
- **当前实现**: 内存向量搜索
- **未来**: FAISS索引优化
- **支持**: 相似度搜索、聚类分析

### 5. SceneRAG
- **功能**: 核心管理类，协调所有组件
- **责任**: 
  - 处理渲染结果
  - 管理语义段数据
  - 执行查询操作
  - 导入导出数据

## 使用方法

### 基础渲染
```bash
./opensplat_render -i model.ply -o output.png -m "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1"
```

### 启用Scene-RAG
```bash
./opensplat_render \
    -i model.ply \
    -o output.png \
    -m "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1" \
    --enable-rag true \
    --rag-output scene_data
```

### 文本查询
```bash
./opensplat_render \
    -i model.ply \
    -o output.png \
    -m "1 0 0 0 0 1 0 0 0 0 1 5 0 0 0 1" \
    --enable-rag true \
    --rag-output scene_data \
    --rag-query "找到桌子和椅子"
```

### 导出场景图
```bash
./opensplat_render \
    --enable-rag true \
    --export-scene-graph scene_graph.json \
    [其他参数...]
```

## 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--enable-rag` | 启用Scene-RAG功能 | false |
| `--rag-output` | Scene-RAG输出基础路径 | "scene_rag" |
| `--rag-query` | 执行文本查询 | - |
| `--export-scene-graph` | 导出场景图JSON文件 | - |

## 输出文件格式

### 向量数据库
- `{output}_vectors.db`: 特征向量和ID映射
- 格式: 文本格式，每行包含segment_id和512维特征

### 元数据
- `{output}_metadata.txt`: 分割区域元数据
- 包含: segment_id, confidence, bbox, gaussian_ids, weights

### 分割掩码
- `{output}_mask_{segment_id}.png`: 各分割区域的二值掩码图像

### 场景图
- JSON格式，包含:
  - segments: 分割区域列表
  - gaussian_mappings: 高斯点到语义段的映射

### 查询结果
- `{output}_query_result_{rank}.png`: 查询结果掩码

## API使用示例

### C++ API
```cpp
#include "scene_rag.hpp"

// 创建Scene-RAG系统
auto scene_rag = std::make_unique<SceneRAG>();

// 处理渲染结果
scene_rag->processRenderResult(rgb_image, px2gid, model, width, height);

// 文本查询
auto results = scene_rag->queryByText("找到红色的物体", 5);

// 图像查询
cv::Mat query_image = cv::imread("reference.jpg");
auto results2 = scene_rag->queryByImage(query_image, cv::Mat(), 5);

// 获取3D高斯点的语义信息
auto semantics = scene_rag->getGaussianSemantics(gaussian_id);
```

### RAG接口
```cpp
#include "scene_rag.hpp"

auto rag_interface = std::make_unique<RAGInterface>(scene_rag);

// 自然语言查询
auto result = rag_interface->queryScene("找到场景中的家具");

// 多模态查询
auto result2 = rag_interface->queryMultimodal(
    "现代风格的椅子", 
    reference_image, 
    reference_mask
);
```

## 数据结构

### SemanticSegment
```cpp
struct SemanticSegment {
    int segment_id;                     // 分割区域ID
    cv::Mat mask;                       // 分割掩码
    cv::Rect bbox;                      // 边界框
    torch::Tensor clip_features;        // CLIP特征向量
    std::vector<int32_t> gaussian_ids;  // 关联的3D高斯点ID
    std::vector<float> gaussian_weights; // 高斯点权重
    std::string semantic_label;         // 语义标签
    float confidence;                   // 分割置信度
};
```

## 性能优化建议

1. **GPU加速**: 确保PyTorch使用CUDA进行特征提取
2. **批处理**: 一次处理多个分割区域的特征提取
3. **索引优化**: 使用FAISS进行大规模向量检索
4. **缓存机制**: 缓存已计算的特征向量
5. **多线程**: 并行处理分割和特征提取

## 扩展和改进

### 短期改进
1. **真实CLIP集成**: 替换简化的特征提取器
2. **SAM集成**: 使用Segment Anything Model进行分割
3. **FAISS索引**: 优化向量数据库性能
4. **GPU内存管理**: 优化大模型内存使用

### 长期发展
1. **多模态融合**: 结合文本、图像、点云信息
2. **时间序列**: 支持视频序列的连续分析
3. **交互式查询**: Web界面和实时查询
4. **知识图谱**: 构建3D场景知识图谱
5. **强化学习**: 基于用户反馈优化查询结果

## 故障排除

### 常见问题

1. **内存不足**
   - 减少图像分辨率
   - 使用CPU模式
   - 增加系统内存

2. **分割效果不佳**
   - 调整confidence_threshold参数
   - 增加min_segment_size过滤小区域
   - 使用更高质量的渲染图像

3. **查询结果不准确**
   - 检查特征提取质量
   - 增加训练数据
   - 调整相似度阈值

4. **编译错误**
   - 确保所有依赖库已安装
   - 检查CMake配置
   - 更新编译器版本

### 调试模式
```bash
# 启用详细输出
export SCENE_RAG_DEBUG=1
./opensplat_render --enable-rag true [其他参数...]
```

## 贡献指南

1. Fork项目仓库
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建Pull Request

## 许可证

本项目遵循AGPLv3许可证，详见LICENSE文件。

## 联系方式

- Issues: 在GitHub仓库中提交问题
- 讨论: 使用GitHub Discussions
- 邮件: [维护者邮箱]

---

**注意**: 当前实现包含简化的分割和特征提取模块，用于演示架构设计。生产环境中建议集成真实的SAM和CLIP模型以获得最佳效果。