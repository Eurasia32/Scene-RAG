#include "scene_rag.hpp"
#include "model_render.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

//============================================================================
// SceneRAG Core Implementation
//============================================================================

SceneRAG::SceneRAG() : next_segment_id_(0) {
    segmentation_module_ = std::make_unique<SegmentationModule>();
    clip_extractor_ = std::make_unique<CLIPFeatureExtractor>();
    vector_db_ = std::make_unique<VectorDatabase>(512);
    mapper_ = std::make_unique<Pixel2GaussianMapper>();
}

SceneRAG::~SceneRAG() = default;

void SceneRAG::processRenderResult(const cv::Mat& rgb_image,
                                 const std::vector<int32_t>* px2gid,
                                 const Model& model,
                                 int image_width, int image_height) {
    
    std::cout << "开始处理Scene-RAG渲染结果..." << std::endl;
    
    // 1. 构建2D-3D映射
    mapper_->buildMapping(px2gid, image_width, image_height);
    
    // 2. 语义分割
    std::cout << "执行语义分割..." << std::endl;
    std::vector<cv::Mat> segment_masks = segmentation_module_->segment(rgb_image);
    std::cout << "生成了 " << segment_masks.size() << " 个分割区域" << std::endl;
    
    // 3. 为每个分割区域提取特征并建立3D映射
    std::vector<torch::Tensor> clip_features;
    std::vector<int> segment_ids;
    
    for (size_t i = 0; i < segment_masks.size(); i++) {
        SemanticSegment segment;
        segment.segment_id = next_segment_id_++;
        segment.mask = segment_masks[i].clone();
        
        // 计算边界框
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(segment.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            segment.bbox = cv::boundingRect(contours[0]);
        }
        
        // 提取CLIP特征
        std::cout << "为分割区域 " << segment.segment_id << " 提取CLIP特征..." << std::endl;
        segment.clip_features = clip_extractor_->extractFeatures(rgb_image, segment.mask);
        
        // 建立2D-3D映射
        mapper_->getGaussianMapping(segment.mask, segment.gaussian_ids, segment.gaussian_weights);
        
        // 计算置信度（基于区域大小和特征质量）
        int pixel_count = cv::countNonZero(segment.mask);
        segment.confidence = std::min(1.0f, pixel_count / 1000.0f);
        
        // 添加到向量数据库
        clip_features.push_back(segment.clip_features);
        segment_ids.push_back(segment.segment_id);
        
        // 更新高斯点到语义段的映射
        for (int32_t gaussian_id : segment.gaussian_ids) {
            gaussian_to_segments_[gaussian_id].push_back(segment.segment_id);
        }
        
        segments_.push_back(std::move(segment));
    }
    
    // 4. 批量添加特征到向量数据库
    if (!clip_features.empty()) {
        vector_db_->addVectors(clip_features, segment_ids);
        std::cout << "已添加 " << clip_features.size() << " 个特征向量到数据库" << std::endl;
    }
    
    std::cout << "Scene-RAG处理完成！" << std::endl;
}

std::vector<SemanticSegment> SceneRAG::queryByText(const std::string& text_query, 
                                                  int top_k) const {
    std::cout << "执行文本查询: \"" << text_query << "\"" << std::endl;
    
    // 简化实现：基于文本生成查询特征（实际需要CLIP文本编码器）
    // 这里使用随机特征作为占位符
    torch::Tensor query_feature = torch::randn({1, 512});
    
    // 在向量数据库中搜索
    auto search_results = vector_db_->search(query_feature, top_k);
    
    std::vector<SemanticSegment> results;
    for (const auto& [segment_id, score] : search_results) {
        // 查找对应的语义段
        auto it = std::find_if(segments_.begin(), segments_.end(),
                              [segment_id](const SemanticSegment& seg) {
                                  return seg.segment_id == segment_id;
                              });
        
        if (it != segments_.end()) {
            SemanticSegment result = *it;
            result.confidence = score;  // 使用搜索得分作为置信度
            results.push_back(result);
        }
    }
    
    std::cout << "返回 " << results.size() << " 个查询结果" << std::endl;
    return results;
}

std::vector<SemanticSegment> SceneRAG::queryByImage(const cv::Mat& query_image,
                                                   const cv::Mat& query_mask,
                                                   int top_k) const {
    std::cout << "执行图像查询..." << std::endl;
    
    // 提取查询图像的特征
    cv::Mat mask = query_mask.empty() ? 
                   cv::Mat::ones(query_image.size(), CV_8UC1) * 255 : 
                   query_mask;
    
    torch::Tensor query_feature = clip_extractor_->extractFeatures(query_image, mask);
    
    // 在向量数据库中搜索
    auto search_results = vector_db_->search(query_feature, top_k);
    
    std::vector<SemanticSegment> results;
    for (const auto& [segment_id, score] : search_results) {
        auto it = std::find_if(segments_.begin(), segments_.end(),
                              [segment_id](const SemanticSegment& seg) {
                                  return seg.segment_id == segment_id;
                              });
        
        if (it != segments_.end()) {
            SemanticSegment result = *it;
            result.confidence = score;
            results.push_back(result);
        }
    }
    
    std::cout << "返回 " << results.size() << " 个图像查询结果" << std::endl;
    return results;
}

std::vector<SemanticSegment> SceneRAG::getGaussianSemantics(int gaussian_id) const {
    std::vector<SemanticSegment> results;
    
    auto it = gaussian_to_segments_.find(gaussian_id);
    if (it != gaussian_to_segments_.end()) {
        for (int segment_id : it->second) {
            auto seg_it = std::find_if(segments_.begin(), segments_.end(),
                                      [segment_id](const SemanticSegment& seg) {
                                          return seg.segment_id == segment_id;
                                      });
            if (seg_it != segments_.end()) {
                results.push_back(*seg_it);
            }
        }
    }
    
    return results;
}

void SceneRAG::exportSceneGraph(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file) {
        std::cerr << "无法创建场景图文件: " << filepath << std::endl;
        return;
    }
    
    // 导出为JSON格式
    file << "{\n";
    file << "  \"segments\": [\n";
    
    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& seg = segments_[i];
        file << "    {\n";
        file << "      \"id\": " << seg.segment_id << ",\n";
        file << "      \"bbox\": [" << seg.bbox.x << ", " << seg.bbox.y 
             << ", " << seg.bbox.width << ", " << seg.bbox.height << "],\n";
        file << "      \"confidence\": " << seg.confidence << ",\n";
        file << "      \"gaussian_count\": " << seg.gaussian_ids.size() << ",\n";
        file << "      \"semantic_label\": \"" << seg.semantic_label << "\"\n";
        file << "    }";
        if (i < segments_.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ],\n";
    file << "  \"gaussian_mappings\": {\n";
    
    bool first = true;
    for (const auto& [gaussian_id, segment_ids] : gaussian_to_segments_) {
        if (!first) file << ",\n";
        file << "    \"" << gaussian_id << "\": [";
        for (size_t i = 0; i < segment_ids.size(); i++) {
            file << segment_ids[i];
            if (i < segment_ids.size() - 1) file << ", ";
        }
        file << "]";
        first = false;
    }
    
    file << "\n  }\n}\n";
    
    std::cout << "场景图已导出到: " << filepath << std::endl;
}

void SceneRAG::save(const std::string& base_filepath) const {
    // 保存向量数据库
    vector_db_->save(base_filepath + "_vectors.db");
    
    // 保存分割掩码和元数据
    std::ofstream meta_file(base_filepath + "_metadata.txt");
    meta_file << segments_.size() << "\n";
    
    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& seg = segments_[i];
        
        // 保存掩码
        std::string mask_path = base_filepath + "_mask_" + std::to_string(seg.segment_id) + ".png";
        cv::imwrite(mask_path, seg.mask);
        
        // 保存元数据
        meta_file << seg.segment_id << " " << seg.confidence << " ";
        meta_file << seg.bbox.x << " " << seg.bbox.y << " " << seg.bbox.width << " " << seg.bbox.height << " ";
        meta_file << seg.gaussian_ids.size() << " ";
        for (int32_t gid : seg.gaussian_ids) {
            meta_file << gid << " ";
        }
        meta_file << seg.gaussian_weights.size() << " ";
        for (float w : seg.gaussian_weights) {
            meta_file << w << " ";
        }
        meta_file << "\n";
    }
    
    std::cout << "Scene-RAG数据已保存到: " << base_filepath << std::endl;
}

void SceneRAG::load(const std::string& base_filepath) {
    // 加载向量数据库
    vector_db_->load(base_filepath + "_vectors.db");
    
    // 加载元数据和掩码
    std::ifstream meta_file(base_filepath + "_metadata.txt");
    if (!meta_file) {
        std::cerr << "无法加载元数据文件" << std::endl;
        return;
    }
    
    int num_segments;
    meta_file >> num_segments;
    
    segments_.clear();
    gaussian_to_segments_.clear();
    
    for (int i = 0; i < num_segments; i++) {
        SemanticSegment seg;
        meta_file >> seg.segment_id >> seg.confidence;
        meta_file >> seg.bbox.x >> seg.bbox.y >> seg.bbox.width >> seg.bbox.height;
        
        int num_gaussians;
        meta_file >> num_gaussians;
        seg.gaussian_ids.resize(num_gaussians);
        for (int j = 0; j < num_gaussians; j++) {
            meta_file >> seg.gaussian_ids[j];
            gaussian_to_segments_[seg.gaussian_ids[j]].push_back(seg.segment_id);
        }
        
        int num_weights;
        meta_file >> num_weights;
        seg.gaussian_weights.resize(num_weights);
        for (int j = 0; j < num_weights; j++) {
            meta_file >> seg.gaussian_weights[j];
        }
        
        // 加载掩码
        std::string mask_path = base_filepath + "_mask_" + std::to_string(seg.segment_id) + ".png";
        seg.mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        
        segments_.push_back(std::move(seg));
    }
    
    next_segment_id_ = segments_.size();
    std::cout << "已加载 " << segments_.size() << " 个语义段" << std::endl;
}

SceneRAG::Statistics SceneRAG::getStatistics() const {
    Statistics stats;
    stats.total_segments = segments_.size();
    stats.total_gaussians_mapped = gaussian_to_segments_.size();
    stats.database_size = vector_db_->size();
    
    // 收集语义标签
    for (const auto& seg : segments_) {
        if (!seg.semantic_label.empty()) {
            stats.semantic_labels.push_back(seg.semantic_label);
        }
    }
    
    return stats;
}

//============================================================================
// RAGInterface Implementation
//============================================================================

RAGInterface::RAGInterface(std::shared_ptr<SceneRAG> scene_rag) 
    : scene_rag_(scene_rag) {}

RAGInterface::QueryResult RAGInterface::queryScene(const std::string& natural_language_query) const {
    QueryResult result;
    
    // 执行查询
    result.segments = scene_rag_->queryByText(natural_language_query, 5);
    
    // 计算得分
    for (const auto& seg : result.segments) {
        result.scores.push_back(seg.confidence);
    }
    
    // 生成解释
    std::ostringstream explanation;
    explanation << "找到 " << result.segments.size() << " 个相关的场景区域，";
    explanation << "基于语义特征匹配查询: \"" << natural_language_query << "\"";
    result.explanation = explanation.str();
    
    return result;
}

RAGInterface::QueryResult RAGInterface::queryMultimodal(const std::string& text_query,
                                                       const cv::Mat& reference_image,
                                                       const cv::Mat& reference_mask) const {
    QueryResult result;
    
    // 结合文本和图像查询
    auto text_results = scene_rag_->queryByText(text_query, 3);
    auto image_results = scene_rag_->queryByImage(reference_image, reference_mask, 3);
    
    // 合并结果（简化实现）
    result.segments = text_results;
    result.segments.insert(result.segments.end(), image_results.begin(), image_results.end());
    
    // 去重
    std::sort(result.segments.begin(), result.segments.end(),
             [](const SemanticSegment& a, const SemanticSegment& b) {
                 return a.segment_id < b.segment_id;
             });
    result.segments.erase(
        std::unique(result.segments.begin(), result.segments.end(),
                   [](const SemanticSegment& a, const SemanticSegment& b) {
                       return a.segment_id == b.segment_id;
                   }),
        result.segments.end());
    
    // 计算得分
    for (const auto& seg : result.segments) {
        result.scores.push_back(seg.confidence);
    }
    
    // 生成解释
    std::ostringstream explanation;
    explanation << "多模态查询结果: " << result.segments.size() << " 个区域，";
    explanation << "结合文本\"" << text_query << "\"和参考图像进行匹配";
    result.explanation = explanation.str();
    
    return result;
}