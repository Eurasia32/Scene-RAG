#ifndef SCENE_RAG_HPP
#define SCENE_RAG_HPP

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>

// 前向声明
class Model;

/**
 * @brief 语义段结构，包含分割区域的所有相关信息
 */
struct SemanticSegment {
    int segment_id;                           // 分割区域ID
    cv::Mat mask;                            // 分割掩码
    cv::Rect bbox;                           // 边界框
    torch::Tensor clip_features;             // CLIP特征向量 (512维)
    std::vector<int32_t> gaussian_ids;       // 关联的3D高斯点ID
    std::vector<float> gaussian_weights;     // 高斯点权重（基于像素贡献度）
    std::string semantic_label;              // 语义标签（可选）
    float confidence;                        // 分割置信度
    
    SemanticSegment() : segment_id(-1), confidence(0.0f) {}
};

/**
 * @brief 2D像素到3D高斯点的映射器
 */
class Pixel2GaussianMapper {
public:
    /**
     * @brief 从px2gid数据构建映射
     */
    void buildMapping(const std::vector<int32_t>* px2gid, int width, int height);
    
    /**
     * @brief 获取分割掩码对应的高斯点ID和权重
     */
    void getGaussianMapping(const cv::Mat& mask, 
                           std::vector<int32_t>& gaussian_ids,
                           std::vector<float>& weights) const;
    
    /**
     * @brief 清理映射数据
     */
    void clear();

private:
    std::vector<std::vector<int32_t>> pixel_to_gaussians;  // 每个像素对应的高斯点ID列表
    int width_, height_;
};

/**
 * @brief 语义分割模块 - 使用SAM (Segment Anything Model)
 */
class SegmentationModule {
public:
    SegmentationModule();
    ~SegmentationModule();
    
    /**
     * @brief 对图像进行语义分割
     */
    std::vector<cv::Mat> segment(const cv::Mat& image);
    
    /**
     * @brief 设置分割参数
     */
    void setParameters(float confidence_threshold = 0.5f, 
                      int min_segment_size = 100);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief CLIP特征提取模块
 */
class CLIPFeatureExtractor {
public:
    CLIPFeatureExtractor();
    ~CLIPFeatureExtractor();
    
    /**
     * @brief 从图像区域提取CLIP特征
     */
    torch::Tensor extractFeatures(const cv::Mat& image, const cv::Mat& mask);
    
    /**
     * @brief 从图像列表批量提取特征
     */
    std::vector<torch::Tensor> extractBatchFeatures(
        const std::vector<cv::Mat>& images,
        const std::vector<cv::Mat>& masks);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 向量数据库接口 - 支持FAISS
 */
class VectorDatabase {
public:
    VectorDatabase(int feature_dim = 512);
    ~VectorDatabase();
    
    /**
     * @brief 添加特征向量
     */
    void addVector(const torch::Tensor& feature, int segment_id);
    
    /**
     * @brief 批量添加特征向量
     */
    void addVectors(const std::vector<torch::Tensor>& features,
                   const std::vector<int>& segment_ids);
    
    /**
     * @brief 相似度搜索
     */
    std::vector<std::pair<int, float>> search(const torch::Tensor& query_feature, 
                                             int top_k = 10) const;
    
    /**
     * @brief 聚类分析
     */
    std::vector<std::vector<int>> cluster(int num_clusters = -1);
    
    /**
     * @brief 保存和加载数据库
     */
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
    
    /**
     * @brief 获取数据库统计信息
     */
    int size() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Scene-RAG核心类
 */
class SceneRAG {
public:
    SceneRAG();
    ~SceneRAG();
    
    /**
     * @brief 处理渲染结果，生成语义段和特征
     */
    void processRenderResult(const cv::Mat& rgb_image,
                           const std::vector<int32_t>* px2gid,
                           const Model& model,
                           int image_width, int image_height);
    
    /**
     * @brief 查询相似场景区域
     */
    std::vector<SemanticSegment> queryByText(const std::string& text_query, 
                                           int top_k = 5) const;
    
    /**
     * @brief 查询相似视觉特征
     */
    std::vector<SemanticSegment> queryByImage(const cv::Mat& query_image,
                                            const cv::Mat& query_mask,
                                            int top_k = 5) const;
    
    /**
     * @brief 获取3D高斯点的语义信息
     */
    std::vector<SemanticSegment> getGaussianSemantics(int gaussian_id) const;
    
    /**
     * @brief 导出语义场景图
     */
    void exportSceneGraph(const std::string& filepath) const;
    
    /**
     * @brief 保存和加载Scene-RAG数据
     */
    void save(const std::string& base_filepath) const;
    void load(const std::string& base_filepath);
    
    /**
     * @brief 获取统计信息
     */
    struct Statistics {
        int total_segments;
        int total_gaussians_mapped;
        int database_size;
        std::vector<std::string> semantic_labels;
    };
    Statistics getStatistics() const;

private:
    std::unique_ptr<SegmentationModule> segmentation_module_;
    std::unique_ptr<CLIPFeatureExtractor> clip_extractor_;
    std::unique_ptr<VectorDatabase> vector_db_;
    std::unique_ptr<Pixel2GaussianMapper> mapper_;
    
    std::vector<SemanticSegment> segments_;
    std::unordered_map<int, std::vector<int>> gaussian_to_segments_;  // 高斯点到语义段的映射
    
    int next_segment_id_;
};

/**
 * @brief RAG查询接口
 */
class RAGInterface {
public:
    RAGInterface(std::shared_ptr<SceneRAG> scene_rag);
    
    /**
     * @brief 自然语言查询
     */
    struct QueryResult {
        std::vector<SemanticSegment> segments;
        std::vector<float> scores;
        std::string explanation;
    };
    
    QueryResult queryScene(const std::string& natural_language_query) const;
    
    /**
     * @brief 多模态查询（文本+图像）
     */
    QueryResult queryMultimodal(const std::string& text_query,
                               const cv::Mat& reference_image,
                               const cv::Mat& reference_mask = cv::Mat()) const;

private:
    std::shared_ptr<SceneRAG> scene_rag_;
};

#endif // SCENE_RAG_HPP