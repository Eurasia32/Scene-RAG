#include "scene_rag.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <fstream>

//============================================================================
// Pixel2GaussianMapper Implementation
//============================================================================

void Pixel2GaussianMapper::buildMapping(const std::vector<int32_t>* px2gid, 
                                       int width, int height) {
    width_ = width;
    height_ = height;
    
    // 清理之前的映射
    pixel_to_gaussians.clear();
    pixel_to_gaussians.resize(width * height);
    
    // 构建像素到高斯点的映射
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            size_t pixIdx = i * width + j;
            pixel_to_gaussians[pixIdx] = px2gid[pixIdx];
        }
    }
    
    std::cout << "构建像素到高斯点映射完成: " << width << "x" << height << std::endl;
}

void Pixel2GaussianMapper::getGaussianMapping(const cv::Mat& mask,
                                            std::vector<int32_t>& gaussian_ids,
                                            std::vector<float>& weights) const {
    if (mask.rows != height_ || mask.cols != width_) {
        throw std::runtime_error("掩码尺寸与映射尺寸不匹配");
    }
    
    std::unordered_map<int32_t, float> gaussian_weight_map;
    int total_pixels = 0;
    
    // 遍历掩码中的每个像素
    for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
            if (mask.at<uint8_t>(i, j) > 0) {  // 像素在掩码内
                size_t pixIdx = i * width_ + j;
                const auto& pixel_gaussians = pixel_to_gaussians[pixIdx];
                
                total_pixels++;
                
                // 为每个贡献的高斯点分配权重
                float weight_per_gaussian = 1.0f / (pixel_gaussians.size() + 1e-6f);
                
                for (int32_t gaussian_id : pixel_gaussians) {
                    gaussian_weight_map[gaussian_id] += weight_per_gaussian;
                }
            }
        }
    }
    
    // 转换为向量并归一化权重
    gaussian_ids.clear();
    weights.clear();
    
    float total_weight = 0.0f;
    for (const auto& [gaussian_id, weight] : gaussian_weight_map) {
        gaussian_ids.push_back(gaussian_id);
        weights.push_back(weight);
        total_weight += weight;
    }
    
    // 归一化权重
    if (total_weight > 1e-6f) {
        for (float& weight : weights) {
            weight /= total_weight;
        }
    }
    
    std::cout << "分割区域映射到 " << gaussian_ids.size() 
              << " 个高斯点，覆盖 " << total_pixels << " 个像素" << std::endl;
}

void Pixel2GaussianMapper::clear() {
    pixel_to_gaussians.clear();
    width_ = height_ = 0;
}

//============================================================================
// SegmentationModule Implementation (简化版本，实际需要集成SAM)
//============================================================================

class SegmentationModule::Impl {
public:
    float confidence_threshold = 0.5f;
    int min_segment_size = 100;
    
    std::vector<cv::Mat> simpleSegmentation(const cv::Mat& image) {
        std::vector<cv::Mat> segments;
        
        // 使用简单的颜色量化进行分割（实际应使用SAM）
        cv::Mat lab_image, segmented;
        cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);
        
        // K-means聚类分割
        cv::Mat data = lab_image.reshape(1, lab_image.rows * lab_image.cols);
        data.convertTo(data, CV_32F);
        
        int K = 8;  // 分割成8个区域
        cv::Mat labels, centers;
        cv::kmeans(data, K, labels,
                  cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
                  3, cv::KMEANS_PP_CENTERS, centers);
        
        // 创建分割掩码
        labels = labels.reshape(0, image.rows);
        
        for (int k = 0; k < K; k++) {
            cv::Mat mask = (labels == k);
            
            // 过滤小区域
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            cv::Mat filtered_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
            for (const auto& contour : contours) {
                if (cv::contourArea(contour) >= min_segment_size) {
                    cv::fillPoly(filtered_mask, std::vector<std::vector<cv::Point>>{contour}, 255);
                }
            }
            
            if (cv::countNonZero(filtered_mask) > 0) {
                segments.push_back(filtered_mask);
            }
        }
        
        return segments;
    }
};

SegmentationModule::SegmentationModule() : impl_(std::make_unique<Impl>()) {}

SegmentationModule::~SegmentationModule() = default;

std::vector<cv::Mat> SegmentationModule::segment(const cv::Mat& image) {
    return impl_->simpleSegmentation(image);
}

void SegmentationModule::setParameters(float confidence_threshold, int min_segment_size) {
    impl_->confidence_threshold = confidence_threshold;
    impl_->min_segment_size = min_segment_size;
}

//============================================================================
// CLIPFeatureExtractor Implementation (简化版本，实际需要集成CLIP)
//============================================================================

class CLIPFeatureExtractor::Impl {
public:
    torch::Tensor extractSimpleFeatures(const cv::Mat& image, const cv::Mat& mask) {
        // 简化版本：使用颜色直方图作为特征（实际应使用CLIP）
        cv::Mat masked_image;
        image.copyTo(masked_image, mask);
        
        // 计算HSV直方图
        cv::Mat hsv;
        cv::cvtColor(masked_image, hsv, cv::COLOR_BGR2HSV);
        
        std::vector<cv::Mat> hsv_planes;
        cv::split(hsv, hsv_planes);
        
        int histSize[] = {50, 60};  // H: 50 bins, S: 60 bins
        float h_ranges[] = {0, 180};
        float s_ranges[] = {0, 256};
        const float* ranges[] = {h_ranges, s_ranges};
        int channels[] = {0, 1};
        
        cv::Mat hist;
        cv::calcHist(&hsv_planes[0], 2, channels, mask, hist, 2, histSize, ranges);
        
        // 归一化直方图
        cv::normalize(hist, hist, 0, 1, cv::NORM_L2);
        
        // 转换为torch tensor (简化为512维)
        std::vector<float> feature_vec;
        hist.reshape(1, hist.total()).copyTo(feature_vec);
        
        // 填充或裁剪到512维
        feature_vec.resize(512, 0.0f);
        
        return torch::from_blob(feature_vec.data(), {1, 512}, torch::kFloat32).clone();
    }
};

CLIPFeatureExtractor::CLIPFeatureExtractor() : impl_(std::make_unique<Impl>()) {}

CLIPFeatureExtractor::~CLIPFeatureExtractor() = default;

torch::Tensor CLIPFeatureExtractor::extractFeatures(const cv::Mat& image, const cv::Mat& mask) {
    return impl_->extractSimpleFeatures(image, mask);
}

std::vector<torch::Tensor> CLIPFeatureExtractor::extractBatchFeatures(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& masks) {
    
    std::vector<torch::Tensor> features;
    for (size_t i = 0; i < images.size() && i < masks.size(); i++) {
        features.push_back(extractFeatures(images[i], masks[i]));
    }
    return features;
}

//============================================================================
// VectorDatabase Implementation (简化版本，实际需要集成FAISS)
//============================================================================

class VectorDatabase::Impl {
public:
    int feature_dim_;
    std::vector<torch::Tensor> features_;
    std::vector<int> segment_ids_;
    
    Impl(int feature_dim) : feature_dim_(feature_dim) {}
    
    void addVector(const torch::Tensor& feature, int segment_id) {
        features_.push_back(feature.clone());
        segment_ids_.push_back(segment_id);
    }
    
    std::vector<std::pair<int, float>> simpleSearch(const torch::Tensor& query_feature, 
                                                   int top_k) const {
        std::vector<std::pair<int, float>> results;
        
        for (size_t i = 0; i < features_.size(); i++) {
            // 计算余弦相似度
            float similarity = torch::cosine_similarity(
                query_feature.flatten(), 
                features_[i].flatten(), 
                0
            ).item<float>();
            
            results.emplace_back(segment_ids_[i], similarity);
        }
        
        // 按相似度排序
        std::sort(results.begin(), results.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 返回top_k结果
        if (results.size() > static_cast<size_t>(top_k)) {
            results.resize(top_k);
        }
        
        return results;
    }
};

VectorDatabase::VectorDatabase(int feature_dim) : impl_(std::make_unique<Impl>(feature_dim)) {}

VectorDatabase::~VectorDatabase() = default;

void VectorDatabase::addVector(const torch::Tensor& feature, int segment_id) {
    impl_->addVector(feature, segment_id);
}

void VectorDatabase::addVectors(const std::vector<torch::Tensor>& features,
                               const std::vector<int>& segment_ids) {
    for (size_t i = 0; i < features.size() && i < segment_ids.size(); i++) {
        addVector(features[i], segment_ids[i]);
    }
}

std::vector<std::pair<int, float>> VectorDatabase::search(const torch::Tensor& query_feature, 
                                                         int top_k) const {
    return impl_->simpleSearch(query_feature, top_k);
}

std::vector<std::vector<int>> VectorDatabase::cluster(int num_clusters) {
    // 简化实现：随机聚类（实际应使用K-means等算法）
    std::vector<std::vector<int>> clusters(num_clusters > 0 ? num_clusters : 3);
    for (size_t i = 0; i < impl_->segment_ids_.size(); i++) {
        int cluster_id = i % clusters.size();
        clusters[cluster_id].push_back(impl_->segment_ids_[i]);
    }
    return clusters;
}

void VectorDatabase::save(const std::string& filepath) const {
    // 简化实现：保存到文本文件
    std::ofstream file(filepath);
    file << impl_->feature_dim_ << "\n";
    file << impl_->features_.size() << "\n";
    
    for (size_t i = 0; i < impl_->features_.size(); i++) {
        file << impl_->segment_ids_[i] << " ";
        auto feature_data = impl_->features_[i].flatten().cpu();
        float* data_ptr = feature_data.data_ptr<float>();
        for (int j = 0; j < impl_->feature_dim_; j++) {
            file << data_ptr[j] << " ";
        }
        file << "\n";
    }
}

void VectorDatabase::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) return;
    
    int feature_dim, num_features;
    file >> feature_dim >> num_features;
    
    impl_->features_.clear();
    impl_->segment_ids_.clear();
    
    for (int i = 0; i < num_features; i++) {
        int segment_id;
        file >> segment_id;
        
        std::vector<float> feature_data(feature_dim);
        for (int j = 0; j < feature_dim; j++) {
            file >> feature_data[j];
        }
        
        torch::Tensor feature = torch::from_blob(feature_data.data(), {1, feature_dim}, torch::kFloat32).clone();
        addVector(feature, segment_id);
    }
}

int VectorDatabase::size() const {
    return impl_->features_.size();
}