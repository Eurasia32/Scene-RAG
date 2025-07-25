#!/usr/bin/env python3
"""
GSRender RAG Backend - Python端RAG后端系统
实现CLIP特征聚类、FAISS索引和文本检索功能
"""

import numpy as np
import torch
import clip
from PIL import Image
import faiss
import pickle
import json
import os
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import logging
from tqdm import tqdm
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GaussianCluster:
    """高斯点聚类结构"""
    cluster_id: int
    gaussian_ids: List[int]
    positions: np.ndarray  # 3D位置 [N, 3]
    features: np.ndarray   # CLIP特征 [N, feature_dim]
    semantic_labels: List[str]  # 语义标签
    confidence_scores: List[float]  # 置信度
    center_position: np.ndarray  # 聚类中心位置 [3]
    center_feature: np.ndarray   # 聚类中心特征 [feature_dim]
    spatial_bounds: Dict[str, float]  # 空间边界 {min_x, max_x, min_y, max_y, min_z, max_z}
    
    def to_dict(self) -> Dict:
        """转换为字典格式（用于序列化）"""
        result = asdict(self)
        # 将numpy数组转换为列表
        result['positions'] = self.positions.tolist()
        result['features'] = self.features.tolist()
        result['center_position'] = self.center_position.tolist()
        result['center_feature'] = self.center_feature.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GaussianCluster':
        """从字典创建对象"""
        data['positions'] = np.array(data['positions'])
        data['features'] = np.array(data['features'])
        data['center_position'] = np.array(data['center_position'])
        data['center_feature'] = np.array(data['center_feature'])
        return cls(**data)

@dataclass
class QueryResult:
    """查询结果结构"""
    cluster_id: int
    similarity_score: float
    gaussian_ids: List[int]
    semantic_labels: List[str]
    confidence_scores: List[float]
    spatial_bounds: Dict[str, float]
    center_position: np.ndarray
    description: str  # 自动生成的描述

class CLIPFeatureExtractor:
    """CLIP特征提取器"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto"):
        """
        初始化CLIP特征提取器
        
        Args:
            model_name: CLIP模型名称
            device: 计算设备
        """
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        logger.info(f"加载CLIP模型: {model_name} on {self.device}")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.feature_dim = self.model.visual.output_dim
        
        logger.info(f"CLIP特征维度: {self.feature_dim}")
    
    def extract_from_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        从图像中提取CLIP特征
        
        Args:
            image: RGB图像 [H, W, 3], 值范围0-1
            mask: 可选掩码 [H, W], 二值
            
        Returns:
            CLIP特征向量 [feature_dim]
        """
        # 转换为PIL图像
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 应用掩码
        if mask is not None:
            masked_image = image.copy()
            masked_image[mask == 0] = 255  # 背景设为白色
            pil_image = Image.fromarray(masked_image)
        else:
            pil_image = Image.fromarray(image)
        
        # 预处理并提取特征
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)  # L2归一化
        
        return features.cpu().numpy().flatten()
    
    def extract_from_text(self, text: str) -> np.ndarray:
        """
        从文本中提取CLIP特征
        
        Args:
            text: 输入文本
            
        Returns:
            CLIP特征向量 [feature_dim]
        """
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)  # L2归一化
        
        return features.cpu().numpy().flatten()
    
    def batch_extract_from_images(self, images: List[np.ndarray], 
                                 masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        批量提取图像特征
        
        Args:
            images: 图像列表
            masks: 可选掩码列表
            
        Returns:
            特征矩阵 [N, feature_dim]
        """
        if masks is None:
            masks = [None] * len(images)
        
        features = []
        for image, mask in tqdm(zip(images, masks), desc="提取CLIP特征", total=len(images)):
            feature = self.extract_from_image(image, mask)
            features.append(feature)
        
        return np.array(features)

class GaussianClusterAnalyzer:
    """高斯点聚类分析器"""
    
    def __init__(self, clip_extractor: CLIPFeatureExtractor):
        """
        初始化聚类分析器
        
        Args:
            clip_extractor: CLIP特征提取器
        """
        self.clip_extractor = clip_extractor
        self.clusters: List[GaussianCluster] = []
        
    def analyze_gaussians_from_renders(self, 
                                     render_results: List[Dict],
                                     camera_poses: List[np.ndarray],
                                     gaussian_positions: np.ndarray,
                                     semantic_threshold: float = 0.7,
                                     spatial_threshold: float = 1.0,
                                     min_cluster_size: int = 5) -> List[GaussianCluster]:
        """
        从多视角渲染结果分析高斯点聚类
        
        Args:
            render_results: 渲染结果列表，每个包含{rgb_image, px2gid_mapping, ...}
            camera_poses: 相机位姿列表
            gaussian_positions: 高斯点3D位置 [N, 3]
            semantic_threshold: 语义相似度阈值
            spatial_threshold: 空间距离阈值
            min_cluster_size: 最小聚类大小
            
        Returns:
            高斯点聚类列表
        """
        logger.info("开始分析高斯点聚类...")
        
        # 1. 收集所有高斯点的CLIP特征
        gaussian_features = self._extract_gaussian_features(
            render_results, camera_poses, gaussian_positions
        )
        
        # 2. 语义聚类
        semantic_clusters = self._semantic_clustering(
            gaussian_features, threshold=semantic_threshold
        )
        
        # 3. 空间聚类
        spatial_clusters = self._spatial_clustering(
            semantic_clusters, gaussian_positions, threshold=spatial_threshold
        )
        
        # 4. 过滤小聚类
        filtered_clusters = [cluster for cluster in spatial_clusters 
                           if len(cluster.gaussian_ids) >= min_cluster_size]
        
        # 5. 生成语义标签
        self._generate_semantic_labels(filtered_clusters, render_results)
        
        self.clusters = filtered_clusters
        logger.info(f"生成了 {len(self.clusters)} 个高斯点聚类")
        
        return self.clusters
    
    def _extract_gaussian_features(self, 
                                  render_results: List[Dict],
                                  camera_poses: List[np.ndarray],
                                  gaussian_positions: np.ndarray) -> Dict[int, List[np.ndarray]]:
        """提取每个高斯点的CLIP特征"""
        logger.info("提取高斯点CLIP特征...")
        
        gaussian_features = {}  # gaussian_id -> [features]
        
        for i, (result, camera_pose) in enumerate(zip(render_results, camera_poses)):
            rgb_image = result['rgb_image']
            px2gid_mapping = result['px2gid_mapping']
            
            logger.info(f"处理视角 {i+1}/{len(render_results)}")
            
            # 为每个像素提取特征
            height, width = rgb_image.shape[:2]
            
            # 使用滑动窗口提取局部特征
            window_size = 32  # 32x32像素窗口
            stride = 16       # 16像素步长
            
            for y in range(0, height - window_size, stride):
                for x in range(0, width - window_size, stride):
                    # 提取窗口区域
                    window = rgb_image[y:y+window_size, x:x+window_size]
                    
                    # 收集该窗口中的高斯点
                    window_gaussians = set()
                    for wy in range(window_size):
                        for wx in range(window_size):
                            pixel_y, pixel_x = y + wy, x + wx
                            if pixel_y < height and pixel_x < width:
                                pixel_idx = pixel_y * width + pixel_x
                                if pixel_idx < len(px2gid_mapping):
                                    window_gaussians.update(px2gid_mapping[pixel_idx])
                    
                    if window_gaussians:
                        # 提取窗口的CLIP特征
                        feature = self.clip_extractor.extract_from_image(window / 255.0)
                        
                        # 将特征分配给窗口中的所有高斯点
                        for gaussian_id in window_gaussians:
                            if gaussian_id not in gaussian_features:
                                gaussian_features[gaussian_id] = []
                            gaussian_features[gaussian_id].append(feature)
        
        # 对每个高斯点的特征进行平均
        averaged_features = {}
        for gaussian_id, features in gaussian_features.items():
            if features:
                averaged_features[gaussian_id] = np.mean(features, axis=0)
        
        logger.info(f"为 {len(averaged_features)} 个高斯点提取了CLIP特征")
        return averaged_features
    
    def _semantic_clustering(self, 
                           gaussian_features: Dict[int, np.ndarray],
                           threshold: float = 0.7) -> List[GaussianCluster]:
        """基于CLIP特征的语义聚类"""
        logger.info("执行语义聚类...")
        
        if not gaussian_features:
            return []
        
        gaussian_ids = list(gaussian_features.keys())
        features = np.array([gaussian_features[gid] for gid in gaussian_ids])
        
        # 计算特征相似度矩阵
        similarity_matrix = cosine_similarity(features)
        
        # 转换为距离矩阵
        distance_matrix = 1 - similarity_matrix
        
        # 使用DBSCAN进行聚类
        dbscan = DBSCAN(eps=1-threshold, min_samples=3, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        # 构建聚类
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
                
            cluster_mask = cluster_labels == label
            cluster_gaussian_ids = [gaussian_ids[i] for i in range(len(gaussian_ids)) if cluster_mask[i]]
            cluster_features = features[cluster_mask]
            
            # 计算质心特征
            center_feature = np.mean(cluster_features, axis=0)
            
            # 创建临时聚类对象（位置信息稍后填充）
            cluster = GaussianCluster(
                cluster_id=len(clusters),
                gaussian_ids=cluster_gaussian_ids,
                positions=np.zeros((len(cluster_gaussian_ids), 3)),  # 临时
                features=cluster_features,
                semantic_labels=[],
                confidence_scores=[],
                center_position=np.zeros(3),  # 临时
                center_feature=center_feature,
                spatial_bounds={}
            )
            
            clusters.append(cluster)
        
        logger.info(f"语义聚类生成了 {len(clusters)} 个聚类")
        return clusters
    
    def _spatial_clustering(self, 
                          semantic_clusters: List[GaussianCluster],
                          gaussian_positions: np.ndarray,
                          threshold: float = 1.0) -> List[GaussianCluster]:
        """基于空间位置的进一步聚类"""
        logger.info("执行空间聚类...")
        
        refined_clusters = []
        
        for semantic_cluster in semantic_clusters:
            # 获取该语义聚类中高斯点的3D位置
            cluster_positions = gaussian_positions[semantic_cluster.gaussian_ids]
            
            if len(cluster_positions) < 3:
                # 太小的聚类直接保留
                semantic_cluster.positions = cluster_positions
                semantic_cluster.center_position = np.mean(cluster_positions, axis=0)
                semantic_cluster.spatial_bounds = self._compute_spatial_bounds(cluster_positions)
                refined_clusters.append(semantic_cluster)
                continue
            
            # 对空间位置进行聚类
            dbscan = DBSCAN(eps=threshold, min_samples=2)
            spatial_labels = dbscan.fit_predict(cluster_positions)
            
            unique_spatial_labels = set(spatial_labels)
            
            for spatial_label in unique_spatial_labels:
                if spatial_label == -1:  # 噪声点，单独处理
                    noise_mask = spatial_labels == -1
                    noise_indices = np.where(noise_mask)[0]
                    
                    for idx in noise_indices:
                        # 每个噪声点单独成一个聚类
                        single_gaussian_id = [semantic_cluster.gaussian_ids[idx]]
                        single_position = cluster_positions[idx:idx+1]
                        single_feature = semantic_cluster.features[idx:idx+1]
                        
                        single_cluster = GaussianCluster(
                            cluster_id=len(refined_clusters),
                            gaussian_ids=single_gaussian_id,
                            positions=single_position,
                            features=single_feature,
                            semantic_labels=[],
                            confidence_scores=[],
                            center_position=single_position[0],
                            center_feature=single_feature[0],
                            spatial_bounds=self._compute_spatial_bounds(single_position)
                        )
                        refined_clusters.append(single_cluster)
                    continue
                
                # 正常的空间聚类
                spatial_mask = spatial_labels == spatial_label
                spatial_gaussian_ids = [semantic_cluster.gaussian_ids[i] 
                                      for i in range(len(semantic_cluster.gaussian_ids)) 
                                      if spatial_mask[i]]
                spatial_positions = cluster_positions[spatial_mask]
                spatial_features = semantic_cluster.features[spatial_mask]
                
                # 创建新的聚类
                new_cluster = GaussianCluster(
                    cluster_id=len(refined_clusters),
                    gaussian_ids=spatial_gaussian_ids,
                    positions=spatial_positions,
                    features=spatial_features,
                    semantic_labels=[],
                    confidence_scores=[],
                    center_position=np.mean(spatial_positions, axis=0),
                    center_feature=np.mean(spatial_features, axis=0),
                    spatial_bounds=self._compute_spatial_bounds(spatial_positions)
                )
                
                refined_clusters.append(new_cluster)
        
        logger.info(f"空间聚类细化后生成了 {len(refined_clusters)} 个聚类")
        return refined_clusters
    
    def _compute_spatial_bounds(self, positions: np.ndarray) -> Dict[str, float]:
        """计算空间边界"""
        if len(positions) == 0:
            return {}
        
        return {
            'min_x': float(np.min(positions[:, 0])),
            'max_x': float(np.max(positions[:, 0])),
            'min_y': float(np.min(positions[:, 1])),
            'max_y': float(np.max(positions[:, 1])),
            'min_z': float(np.min(positions[:, 2])),
            'max_z': float(np.max(positions[:, 2]))
        }
    
    def _generate_semantic_labels(self, 
                                clusters: List[GaussianCluster],
                                render_results: List[Dict]) -> None:
        """为聚类生成语义标签"""
        logger.info("生成语义标签...")
        
        # 预定义的语义标签
        semantic_candidates = [
            "furniture", "table", "chair", "sofa", "bed", "cabinet",
            "wall", "floor", "ceiling", "window", "door",
            "decoration", "plant", "lamp", "book", "electronic device",
            "kitchen appliance", "bathroom fixture", "artwork", "mirror"
        ]
        
        # 为每个候选标签提取CLIP特征
        candidate_features = {}
        for label in semantic_candidates:
            candidate_features[label] = self.clip_extractor.extract_from_text(label)
        
        # 为每个聚类分配最匹配的语义标签
        for cluster in clusters:
            # 计算聚类中心特征与候选标签的相似度
            similarities = {}
            for label, label_feature in candidate_features.items():
                similarity = cosine_similarity(
                    cluster.center_feature.reshape(1, -1),
                    label_feature.reshape(1, -1)
                )[0, 0]
                similarities[label] = similarity
            
            # 选择最相似的前3个标签
            top_labels = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
            
            cluster.semantic_labels = [label for label, _ in top_labels]
            cluster.confidence_scores = [score for _, score in top_labels]

class FAISSVectorIndex:
    """FAISS向量索引"""
    
    def __init__(self, feature_dim: int, index_type: str = "IVF"):
        """
        初始化FAISS索引
        
        Args:
            feature_dim: 特征维度
            index_type: 索引类型 ("Flat", "IVF", "HNSW")
        """
        self.feature_dim = feature_dim
        self.index_type = index_type
        self.index = None
        self.cluster_metadata = {}  # cluster_id -> GaussianCluster
        
        logger.info(f"初始化FAISS索引: {index_type}, 特征维度: {feature_dim}")
    
    def build_index(self, clusters: List[GaussianCluster]) -> None:
        """
        构建FAISS索引
        
        Args:
            clusters: 高斯点聚类列表
        """
        if not clusters:
            logger.warning("没有聚类数据，无法构建索引")
            return
        
        logger.info(f"构建FAISS索引，聚类数: {len(clusters)}")
        
        # 准备特征矩阵
        features = []
        cluster_ids = []
        
        for cluster in clusters:
            features.append(cluster.center_feature)
            cluster_ids.append(cluster.cluster_id)
            self.cluster_metadata[cluster.cluster_id] = cluster
        
        features = np.array(features).astype('float32')
        
        # 创建FAISS索引
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.feature_dim)  # 内积相似度
        elif self.index_type == "IVF":
            nlist = min(100, len(clusters) // 4)  # 聚类中心数
            quantizer = faiss.IndexFlatIP(self.feature_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.feature_dim, nlist)
            
            # 训练索引
            self.index.train(features)
        elif self.index_type == "HNSW":
            M = 32  # 连接数
            self.index = faiss.IndexHNSWFlat(self.feature_dim, M)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        # 添加向量到索引
        self.index.add(features)
        
        logger.info(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")
    
    def search(self, query_feature: np.ndarray, top_k: int = 10) -> List[QueryResult]:
        """
        搜索最相似的聚类
        
        Args:
            query_feature: 查询特征向量
            top_k: 返回最相似的前k个结果
            
        Returns:
            查询结果列表
        """
        if self.index is None:
            raise RuntimeError("索引未构建，请先调用build_index()")
        
        query_feature = query_feature.astype('float32').reshape(1, -1)
        
        # 执行搜索
        similarities, indices = self.index.search(query_feature, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            similarity = similarities[0][i]
            
            if idx in self.cluster_metadata:
                cluster = self.cluster_metadata[idx]
                
                # 生成描述
                description = self._generate_description(cluster)
                
                result = QueryResult(
                    cluster_id=cluster.cluster_id,
                    similarity_score=float(similarity),
                    gaussian_ids=cluster.gaussian_ids,
                    semantic_labels=cluster.semantic_labels,
                    confidence_scores=cluster.confidence_scores,
                    spatial_bounds=cluster.spatial_bounds,
                    center_position=cluster.center_position,
                    description=description
                )
                
                results.append(result)
        
        return results
    
    def _generate_description(self, cluster: GaussianCluster) -> str:
        """生成聚类描述"""
        if not cluster.semantic_labels:
            return f"聚类 {cluster.cluster_id}: {len(cluster.gaussian_ids)} 个高斯点"
        
        primary_label = cluster.semantic_labels[0]
        confidence = cluster.confidence_scores[0] if cluster.confidence_scores else 0.0
        
        # 空间信息
        bounds = cluster.spatial_bounds
        if bounds:
            size_x = bounds.get('max_x', 0) - bounds.get('min_x', 0)
            size_y = bounds.get('max_y', 0) - bounds.get('min_y', 0)
            size_z = bounds.get('max_z', 0) - bounds.get('min_z', 0)
            volume = size_x * size_y * size_z
            
            description = (f"{primary_label} (置信度: {confidence:.2f}) - "
                         f"{len(cluster.gaussian_ids)} 个高斯点, "
                         f"空间体积: {volume:.2f}m³")
        else:
            description = (f"{primary_label} (置信度: {confidence:.2f}) - "
                         f"{len(cluster.gaussian_ids)} 个高斯点")
        
        return description
    
    def save(self, filepath: str) -> None:
        """保存索引和元数据"""
        if self.index is None:
            raise RuntimeError("索引未构建")
        
        # 保存FAISS索引
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # 保存元数据
        metadata = {
            'feature_dim': self.feature_dim,
            'index_type': self.index_type,
            'clusters': {cid: cluster.to_dict() for cid, cluster in self.cluster_metadata.items()}
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"索引已保存到 {filepath}.faiss 和 {filepath}.pkl")
    
    def load(self, filepath: str) -> None:
        """加载索引和元数据"""
        # 加载FAISS索引
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # 加载元数据
        with open(f"{filepath}.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_dim = metadata['feature_dim']
        self.index_type = metadata['index_type']
        
        # 重建聚类元数据
        self.cluster_metadata = {}
        for cid, cluster_data in metadata['clusters'].items():
            self.cluster_metadata[int(cid)] = GaussianCluster.from_dict(cluster_data)
        
        logger.info(f"索引已从 {filepath} 加载，包含 {len(self.cluster_metadata)} 个聚类")

class GSRenderRAGBackend:
    """GSRender RAG后端主类"""
    
    def __init__(self, clip_model: str = "ViT-B/32", device: str = "auto"):
        """
        初始化RAG后端
        
        Args:
            clip_model: CLIP模型名称
            device: 计算设备
        """
        logger.info("初始化GSRender RAG后端...")
        
        self.clip_extractor = CLIPFeatureExtractor(clip_model, device)
        self.cluster_analyzer = GaussianClusterAnalyzer(self.clip_extractor)
        self.vector_index = FAISSVectorIndex(self.clip_extractor.feature_dim)
        
        self.is_built = False
        
        logger.info("RAG后端初始化完成")
    
    def build_rag_from_renders(self,
                              gsrender_interface,
                              camera_poses: List[np.ndarray],
                              **clustering_params) -> None:
        """
        从渲染结果构建RAG系统
        
        Args:
            gsrender_interface: GSRender接口对象
            camera_poses: 相机位姿列表
            **clustering_params: 聚类参数
        """
        logger.info("从渲染结果构建RAG系统...")
        
        # 获取模型信息
        model_info = gsrender_interface.get_model_info()
        gaussian_positions = model_info.means  # [N, 3]
        
        logger.info(f"处理 {len(gaussian_positions)} 个高斯点，{len(camera_poses)} 个视角")
        
        # 渲染多个视角
        render_results = []
        for i, camera_pose in enumerate(tqdm(camera_poses, desc="渲染视角")):
            camera_params = self._create_camera_params(camera_pose)
            result = gsrender_interface.render(camera_params)
            
            render_data = {
                'rgb_image': result.rgb_image,
                'px2gid_mapping': result.pixel_to_gaussian_mapping,
                'camera_pose': camera_pose
            }
            render_results.append(render_data)
        
        # 分析高斯点聚类
        clusters = self.cluster_analyzer.analyze_gaussians_from_renders(
            render_results, camera_poses, gaussian_positions, **clustering_params
        )
        
        # 构建FAISS索引
        self.vector_index.build_index(clusters)
        
        self.is_built = True
        logger.info("RAG系统构建完成")
    
    def query(self, text: str, top_k: int = 5) -> List[QueryResult]:
        """
        文本查询
        
        Args:
            text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            查询结果列表
        """
        if not self.is_built:
            raise RuntimeError("RAG系统未构建，请先调用build_rag_from_renders()")
        
        logger.info(f"执行文本查询: '{text}'")
        
        # 提取文本特征
        query_feature = self.clip_extractor.extract_from_text(text)
        
        # 搜索相似聚类
        results = self.vector_index.search(query_feature, top_k)
        
        logger.info(f"找到 {len(results)} 个匹配结果")
        return results
    
    def query_by_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None, 
                      top_k: int = 5) -> List[QueryResult]:
        """
        图像查询
        
        Args:
            image: 查询图像
            mask: 可选掩码
            top_k: 返回结果数量
            
        Returns:
            查询结果列表
        """
        if not self.is_built:
            raise RuntimeError("RAG系统未构建")
        
        logger.info("执行图像查询")
        
        # 提取图像特征
        query_feature = self.clip_extractor.extract_from_image(image, mask)
        
        # 搜索相似聚类
        results = self.vector_index.search(query_feature, top_k)
        
        return results
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """获取聚类统计信息"""
        if not self.is_built:
            return {"error": "RAG系统未构建"}
        
        clusters = list(self.vector_index.cluster_metadata.values())
        
        total_gaussians = sum(len(cluster.gaussian_ids) for cluster in clusters)
        semantic_labels = {}
        
        for cluster in clusters:
            for label in cluster.semantic_labels:
                semantic_labels[label] = semantic_labels.get(label, 0) + 1
        
        return {
            "total_clusters": len(clusters),
            "total_gaussians": total_gaussians,
            "avg_gaussians_per_cluster": total_gaussians / len(clusters) if clusters else 0,
            "semantic_label_counts": semantic_labels,
            "feature_dimension": self.clip_extractor.feature_dim
        }
    
    def save(self, filepath: str) -> None:
        """保存RAG系统"""
        if not self.is_built:
            raise RuntimeError("RAG系统未构建")
        
        self.vector_index.save(filepath)
        logger.info(f"RAG系统已保存到 {filepath}")
    
    def load(self, filepath: str) -> None:
        """加载RAG系统"""
        self.vector_index.load(filepath)
        self.is_built = True
        logger.info(f"RAG系统已从 {filepath} 加载")
    
    def _create_camera_params(self, camera_pose: np.ndarray):
        """创建相机参数（需要根据实际的gsrender接口调整）"""
        # 这里需要根据实际的gsrender.create_camera_params接口进行调整
        import gsrender
        return gsrender.create_camera_params(
            view_matrix=camera_pose.astype(np.float32),
            width=640,
            height=480,
            fx=320.0,
            fy=320.0,
            downscale_factor=1.0
        )

# 导出主要类
__all__ = [
    'GSRenderRAGBackend',
    'GaussianCluster', 
    'QueryResult',
    'CLIPFeatureExtractor',
    'GaussianClusterAnalyzer',
    'FAISSVectorIndex'
]