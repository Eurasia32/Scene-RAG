#!/usr/bin/env python3
"""
智能化RAG加速系统 - LLM查询意图分解与多因子重排序
实现动态数据库式的高性能3D场景检索
"""

import json
import numpy as np
import torch
import asyncio
import aiohttp
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod
import openai
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import faiss

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    OBJECT_SEARCH = "object_search"        # 对象搜索
    SPATIAL_QUERY = "spatial_query"        # 空间查询
    VISUAL_SEARCH = "visual_search"        # 视觉搜索
    SEMANTIC_QUERY = "semantic_query"      # 语义查询
    COMPOSITE_QUERY = "composite_query"    # 复合查询

@dataclass
class QueryIntent:
    """查询意图结构"""
    query_type: QueryType
    primary_objects: List[str]              # 主要查询对象
    secondary_objects: List[str]            # 次要对象
    spatial_constraints: Dict[str, Any]     # 空间约束
    visual_attributes: Dict[str, Any]       # 视觉属性
    semantic_context: Dict[str, Any]        # 语义上下文
    confidence: float                       # 意图解析置信度
    priority_weights: Dict[str, float]      # 各因子权重
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = asdict(self)
        result['query_type'] = self.query_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QueryIntent':
        """从字典创建"""
        data['query_type'] = QueryType(data['query_type'])
        return cls(**data)

class LLMProvider(ABC):
    """LLM提供商抽象基类"""
    
    @abstractmethod
    async def analyze_query_intent(self, query: str, context: Dict = None) -> QueryIntent:
        """分析查询意图"""
        pass

class OpenAICompatibleProvider(LLMProvider):
    """OpenAI兼容API提供商（支持OpenAI、Azure OpenAI、vLLM、Ollama等）"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url
    
    async def analyze_query_intent(self, query: str, context: Dict = None) -> QueryIntent:
        """使用GPT分析查询意图"""
        
        system_prompt = """你是一个3D场景查询意图分析专家。请将用户的自然语言查询转换为结构化的JSON格式。

分析以下查询并提取：
1. 查询类型：object_search, spatial_query, visual_search, semantic_query, composite_query
2. 主要对象：用户最关心的物体
3. 次要对象：相关的背景物体
4. 空间约束：位置、方向、距离等
5. 视觉属性：颜色、形状、材质等
6. 语义上下文：功能、用途、场景等
7. 优先级权重：各个匹配因子的重要程度

请返回以下JSON格式：
{
  "query_type": "object_search",
  "primary_objects": ["chair", "table"],
  "secondary_objects": ["room", "furniture"],
  "spatial_constraints": {
    "location": "center",
    "proximity": "near window",
    "orientation": "facing door",
    "bounds": {"x": [-2, 2], "y": [0, 1], "z": [-2, 2]}
  },
  "visual_attributes": {
    "color": ["red", "brown"],
    "material": ["wood", "leather"],
    "style": ["modern", "minimalist"]
  },
  "semantic_context": {
    "function": "seating",
    "scene_type": "living_room",
    "usage": "dining"
  },
  "confidence": 0.95,
  "priority_weights": {
    "vector_similarity": 0.3,
    "text_similarity": 0.2,
    "visual_similarity": 0.2,
    "spatial_relevance": 0.2,
    "multi_view_consistency": 0.1
  }
}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"查询: {query}"}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # 解析JSON响应
            content = response.choices[0].message.content
            
            # 提取JSON部分
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                intent_data = json.loads(json_str)
                return QueryIntent.from_dict(intent_data)
            else:
                raise ValueError("无法从LLM响应中提取JSON")
                
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            # 返回默认意图
            return self._create_fallback_intent(query)
    
    def _create_fallback_intent(self, query: str) -> QueryIntent:
        """创建回退意图"""
        return QueryIntent(
            query_type=QueryType.OBJECT_SEARCH,
            primary_objects=[query.split()[0] if query.split() else "object"],
            secondary_objects=[],
            spatial_constraints={},
            visual_attributes={},
            semantic_context={},
            confidence=0.5,
            priority_weights={
                "vector_similarity": 0.4,
                "text_similarity": 0.3,
                "visual_similarity": 0.1,
                "spatial_relevance": 0.1,
                "multi_view_consistency": 0.1
            }
        )

class GaussianPruner:
    """基于查询意图的高斯点剪枝器"""
    
    def __init__(self, model_info, spatial_index=None):
        """
        初始化剪枝器
        
        Args:
            model_info: 模型信息（位置、特征等）
            spatial_index: 空间索引（如KDTree）
        """
        self.gaussian_positions = model_info.means  # [N, 3]
        self.gaussian_features = getattr(model_info, 'features', None)
        self.num_gaussians = len(self.gaussian_positions)
        
        # 构建空间索引
        self.spatial_index = self._build_spatial_index()
        
        # 预计算的语义映射（简化版）
        self.semantic_labels = self._precompute_semantic_labels()
        
        logger.info(f"高斯点剪枝器初始化完成，包含 {self.num_gaussians} 个高斯点")
    
    def _build_spatial_index(self):
        """构建空间索引"""
        try:
            from sklearn.neighbors import KDTree
            return KDTree(self.gaussian_positions)
        except ImportError:
            logger.warning("sklearn不可用，使用简单空间索引")
            return None
    
    def _precompute_semantic_labels(self) -> Dict[int, List[str]]:
        """预计算语义标签（简化版）"""
        # 这里应该使用预训练的模型或预计算的结果
        # 简化实现：基于位置推断语义
        semantic_map = {}
        
        for i, pos in enumerate(self.gaussian_positions):
            x, y, z = pos
            labels = []
            
            # 基于高度推断
            if y < 0.1:
                labels.append('floor')
            elif y > 2.0:
                labels.append('ceiling')
            elif 0.5 < y < 1.5:
                labels.append('furniture')
            
            # 基于位置推断
            if abs(x) > 2 or abs(z) > 2:
                labels.append('wall')
            else:
                labels.append('interior')
            
            semantic_map[i] = labels
        
        return semantic_map
    
    def prune_by_intent(self, intent: QueryIntent, 
                       downsample_factor: float = 0.3) -> List[int]:
        """
        基于查询意图剪枝高斯点
        
        Args:
            intent: 查询意图
            downsample_factor: 降采样因子
            
        Returns:
            保留的高斯点索引列表
        """
        logger.info(f"开始基于意图剪枝，目标保留 {downsample_factor*100:.1f}% 的高斯点")
        
        # 第一阶段：粗粒度过滤
        candidates = self._coarse_filtering(intent)
        
        # 第二阶段：精细化过滤
        refined_candidates = self._fine_filtering(candidates, intent)
        
        # 第三阶段：动态采样
        final_indices = self._dynamic_sampling(
            refined_candidates, intent, downsample_factor
        )
        
        logger.info(f"剪枝完成：{self.num_gaussians} → {len(final_indices)} "
                   f"({len(final_indices)/self.num_gaussians*100:.1f}%)")
        
        return final_indices
    
    def _coarse_filtering(self, intent: QueryIntent) -> List[int]:
        """粗粒度过滤"""
        candidates = set(range(self.num_gaussians))
        
        # 空间约束过滤
        if intent.spatial_constraints:
            spatial_candidates = self._filter_by_spatial_constraints(
                intent.spatial_constraints
            )
            candidates = candidates.intersection(spatial_candidates)
        
        # 语义标签过滤
        if intent.primary_objects:
            semantic_candidates = self._filter_by_semantic_labels(
                intent.primary_objects
            )
            candidates = candidates.intersection(semantic_candidates)
        
        return list(candidates)
    
    def _filter_by_spatial_constraints(self, constraints: Dict) -> set:
        """基于空间约束过滤"""
        valid_indices = set()
        
        # 边界框约束
        if 'bounds' in constraints:
            bounds = constraints['bounds']
            for i, pos in enumerate(self.gaussian_positions):
                x, y, z = pos
                if (bounds.get('x', [-float('inf'), float('inf')])[0] <= x <= 
                    bounds.get('x', [-float('inf'), float('inf')])[1] and
                    bounds.get('y', [-float('inf'), float('inf')])[0] <= y <= 
                    bounds.get('y', [-float('inf'), float('inf')])[1] and
                    bounds.get('z', [-float('inf'), float('inf')])[0] <= z <= 
                    bounds.get('z', [-float('inf'), float('inf')])[1]):
                    valid_indices.add(i)
        else:
            valid_indices = set(range(self.num_gaussians))
        
        # 位置关键词约束
        if 'location' in constraints:
            location = constraints['location']
            location_indices = self._filter_by_location_keyword(location)
            valid_indices = valid_indices.intersection(location_indices)
        
        return valid_indices
    
    def _filter_by_location_keyword(self, location: str) -> set:
        """基于位置关键词过滤"""
        valid_indices = set()
        
        if location == 'center':
            # 中心区域
            center = np.mean(self.gaussian_positions, axis=0)
            distances = np.linalg.norm(self.gaussian_positions - center, axis=1)
            threshold = np.percentile(distances, 50)  # 中心50%
            valid_indices = set(np.where(distances <= threshold)[0])
        
        elif location == 'corner':
            # 角落区域
            center = np.mean(self.gaussian_positions, axis=0)
            distances = np.linalg.norm(self.gaussian_positions - center, axis=1)
            threshold = np.percentile(distances, 80)  # 远离中心80%
            valid_indices = set(np.where(distances >= threshold)[0])
        
        else:
            # 默认不过滤
            valid_indices = set(range(self.num_gaussians))
        
        return valid_indices
    
    def _filter_by_semantic_labels(self, target_objects: List[str]) -> set:
        """基于语义标签过滤"""
        valid_indices = set()
        
        for idx, labels in self.semantic_labels.items():
            # 检查是否匹配任何目标对象
            for obj in target_objects:
                if any(obj.lower() in label.lower() for label in labels):
                    valid_indices.add(idx)
                    break
        
        return valid_indices
    
    def _fine_filtering(self, candidates: List[int], intent: QueryIntent) -> List[int]:
        """精细化过滤"""
        if not candidates:
            return candidates
        
        # 基于视觉属性进一步过滤
        if intent.visual_attributes and self.gaussian_features is not None:
            # 这里可以基于预计算的视觉特征进行过滤
            pass
        
        # 基于语义上下文过滤
        if intent.semantic_context:
            # 场景类型过滤
            if 'scene_type' in intent.semantic_context:
                scene_type = intent.semantic_context['scene_type']
                candidates = self._filter_by_scene_type(candidates, scene_type)
        
        return candidates
    
    def _filter_by_scene_type(self, candidates: List[int], scene_type: str) -> List[int]:
        """基于场景类型过滤"""
        # 简化实现
        if scene_type == 'living_room':
            # 客厅场景：保留中等高度的点
            valid_candidates = []
            for idx in candidates:
                y = self.gaussian_positions[idx][1]
                if 0.2 <= y <= 2.0:  # 家具高度范围
                    valid_candidates.append(idx)
            return valid_candidates
        
        return candidates
    
    def _dynamic_sampling(self, candidates: List[int], intent: QueryIntent, 
                         target_ratio: float) -> List[int]:
        """动态采样"""
        if not candidates:
            return candidates
        
        target_count = max(1, int(len(candidates) * target_ratio))
        
        if len(candidates) <= target_count:
            return candidates
        
        # 基于重要性采样
        importance_scores = self._compute_importance_scores(candidates, intent)
        
        # 选择重要性最高的点
        sorted_indices = np.argsort(importance_scores)[::-1]
        selected_indices = [candidates[i] for i in sorted_indices[:target_count]]
        
        return selected_indices
    
    def _compute_importance_scores(self, candidates: List[int], 
                                  intent: QueryIntent) -> np.ndarray:
        """计算重要性分数"""
        scores = np.ones(len(candidates))
        
        # 基于位置的重要性
        positions = self.gaussian_positions[candidates]
        
        # 距离中心的远近
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        
        # 根据查询类型调整重要性
        if intent.query_type == QueryType.SPATIAL_QUERY:
            # 空间查询更关注位置分布
            scores = 1.0 / (1.0 + distances)
        else:
            # 对象搜索更关注密度
            scores = np.exp(-distances / np.std(distances))
        
        return scores

@dataclass
class MultiFactorScore:
    """多因子评分结果"""
    cluster_id: int
    vector_similarity: float      # 向量相似度
    text_similarity: float        # 文本相似度  
    visual_similarity: float      # 视觉相似度
    spatial_relevance: float      # 空间相关性
    multi_view_consistency: float # 多视角一致性
    final_score: float           # 最终得分
    
    def to_dict(self) -> Dict:
        return asdict(self)

class MultiFactorReranker:
    """多因子重排序器"""
    
    def __init__(self, clip_extractor, vector_index):
        """
        初始化重排序器
        
        Args:
            clip_extractor: CLIP特征提取器
            vector_index: 向量索引
        """
        self.clip_extractor = clip_extractor
        self.vector_index = vector_index
        self.scaler = MinMaxScaler()
        
        logger.info("多因子重排序器初始化完成")
    
    def rerank_results(self, query: str, intent: QueryIntent, 
                      initial_results: List, top_k: int = 10) -> List[MultiFactorScore]:
        """
        多因子重排序
        
        Args:
            query: 原始查询
            intent: 查询意图
            initial_results: 初始结果列表
            top_k: 最终返回的结果数量
            
        Returns:
            重排序后的结果列表
        """
        logger.info(f"开始多因子重排序，输入 {len(initial_results)} 个结果")
        
        if not initial_results:
            return []
        
        # 计算各个因子的相似度
        factor_scores = []
        
        for result in initial_results:
            cluster_id = result.cluster_id
            
            # 1. 向量相似度（已有）
            vector_sim = result.similarity_score
            
            # 2. 文本相似度
            text_sim = self._compute_text_similarity(query, result, intent)
            
            # 3. 视觉相似度
            visual_sim = self._compute_visual_similarity(query, result, intent)
            
            # 4. 空间相关性
            spatial_rel = self._compute_spatial_relevance(result, intent)
            
            # 5. 多视角一致性
            multi_view = self._compute_multi_view_consistency(result)
            
            factor_scores.append({
                'cluster_id': cluster_id,
                'vector_similarity': vector_sim,
                'text_similarity': text_sim,
                'visual_similarity': visual_sim,
                'spatial_relevance': spatial_rel,
                'multi_view_consistency': multi_view
            })
        
        # 归一化各因子分数
        normalized_scores = self._normalize_scores(factor_scores)
        
        # 加权计算最终分数
        final_scores = self._compute_weighted_scores(normalized_scores, intent)
        
        # 排序并返回top-k
        sorted_scores = sorted(final_scores, key=lambda x: x.final_score, reverse=True)
        
        logger.info(f"重排序完成，返回前 {min(top_k, len(sorted_scores))} 个结果")
        return sorted_scores[:top_k]
    
    def _compute_text_similarity(self, query: str, result, intent: QueryIntent) -> float:
        """计算文本相似度"""
        # 提取结果的文本描述
        result_text = result.description if hasattr(result, 'description') else ""
        result_labels = result.semantic_labels if hasattr(result, 'semantic_labels') else []
        
        # 组合文本
        combined_text = result_text + " " + " ".join(result_labels)
        
        if not combined_text.strip():
            return 0.0
        
        # 使用CLIP计算文本相似度
        query_features = self.clip_extractor.extract_from_text(query)
        result_features = self.clip_extractor.extract_from_text(combined_text)
        
        # 余弦相似度
        similarity = np.dot(query_features, result_features) / (
            np.linalg.norm(query_features) * np.linalg.norm(result_features)
        )
        
        return float(similarity)
    
    def _compute_visual_similarity(self, query: str, result, intent: QueryIntent) -> float:
        """计算视觉相似度"""
        # 基于视觉属性的相似度
        visual_attrs = intent.visual_attributes
        
        if not visual_attrs:
            return 0.5  # 中性分数
        
        similarity_score = 0.0
        total_factors = 0
        
        # 颜色匹配
        if 'colors' in visual_attrs and visual_attrs['colors']:
            color_match = self._match_visual_attribute(
                result, 'color', visual_attrs['colors']
            )
            similarity_score += color_match
            total_factors += 1
        
        # 材质匹配
        if 'material' in visual_attrs and visual_attrs['material']:
            material_match = self._match_visual_attribute(
                result, 'material', visual_attrs['material']
            )
            similarity_score += material_match
            total_factors += 1
        
        # 风格匹配
        if 'style' in visual_attrs and visual_attrs['style']:
            style_match = self._match_visual_attribute(
                result, 'style', visual_attrs['style']
            )
            similarity_score += style_match
            total_factors += 1
        
        return similarity_score / max(1, total_factors)
    
    def _match_visual_attribute(self, result, attr_type: str, target_values: List[str]) -> float:
        """匹配视觉属性"""
        # 简化实现：基于语义标签匹配
        result_labels = getattr(result, 'semantic_labels', [])
        
        matches = 0
        for target in target_values:
            for label in result_labels:
                if target.lower() in label.lower():
                    matches += 1
                    break
        
        return matches / max(1, len(target_values))
    
    def _compute_spatial_relevance(self, result, intent: QueryIntent) -> float:
        """计算空间相关性"""
        spatial_constraints = intent.spatial_constraints
        
        if not spatial_constraints:
            return 0.5  # 中性分数
        
        spatial_score = 0.0
        total_factors = 0
        
        # 位置约束
        if 'location' in spatial_constraints:
            location_score = self._evaluate_location_constraint(
                result, spatial_constraints['location']
            )
            spatial_score += location_score
            total_factors += 1
        
        # 边界约束
        if 'bounds' in spatial_constraints:
            bounds_score = self._evaluate_bounds_constraint(
                result, spatial_constraints['bounds']
            )
            spatial_score += bounds_score
            total_factors += 1
        
        # 邻近性约束
        if 'proximity' in spatial_constraints:
            proximity_score = self._evaluate_proximity_constraint(
                result, spatial_constraints['proximity']
            )
            spatial_score += proximity_score
            total_factors += 1
        
        return spatial_score / max(1, total_factors)
    
    def _evaluate_location_constraint(self, result, location: str) -> float:
        """评估位置约束"""
        # 获取结果的空间位置
        if hasattr(result, 'center_position'):
            position = result.center_position
        elif hasattr(result, 'spatial_bounds'):
            bounds = result.spatial_bounds
            position = np.array([
                (bounds.get('min_x', 0) + bounds.get('max_x', 0)) / 2,
                (bounds.get('min_y', 0) + bounds.get('max_y', 0)) / 2,
                (bounds.get('min_z', 0) + bounds.get('max_z', 0)) / 2
            ])
        else:
            return 0.5
        
        # 基于位置关键词评分
        if location == 'center':
            # 计算到中心的距离（假设场景中心为原点）
            distance = np.linalg.norm(position)
            return max(0, 1 - distance / 5.0)  # 5米为最大有效距离
        
        elif location == 'corner':
            # 角落位置：距离中心较远
            distance = np.linalg.norm(position)
            return min(1, distance / 3.0)  # 3米以上为角落
        
        return 0.5
    
    def _evaluate_bounds_constraint(self, result, bounds: Dict) -> float:
        """评估边界约束"""
        if not hasattr(result, 'spatial_bounds'):
            return 0.5
        
        result_bounds = result.spatial_bounds
        
        # 计算边界重叠度
        overlap_score = 0.0
        dimensions = ['x', 'y', 'z']
        
        for dim in dimensions:
            if dim in bounds:
                target_min, target_max = bounds[dim]
                result_min = result_bounds.get(f'min_{dim}', target_min)
                result_max = result_bounds.get(f'max_{dim}', target_max)
                
                # 计算重叠
                overlap_min = max(target_min, result_min)
                overlap_max = min(target_max, result_max)
                
                if overlap_max > overlap_min:
                    overlap_size = overlap_max - overlap_min
                    target_size = target_max - target_min
                    overlap_ratio = overlap_size / max(target_size, 1e-6)
                    overlap_score += overlap_ratio
        
        return overlap_score / len(dimensions)
    
    def _evaluate_proximity_constraint(self, result, proximity: str) -> float:
        """评估邻近性约束"""
        # 简化实现：基于描述匹配
        if hasattr(result, 'description'):
            description = result.description.lower()
            proximity_lower = proximity.lower()
            
            # 简单的关键词匹配
            if proximity_lower in description:
                return 1.0
            
            # 部分匹配
            proximity_words = proximity_lower.split()
            matches = sum(1 for word in proximity_words if word in description)
            return matches / max(1, len(proximity_words))
        
        return 0.5
    
    def _compute_multi_view_consistency(self, result) -> float:
        """计算多视角一致性"""
        # 简化实现：基于高斯点数量和空间分布
        if hasattr(result, 'gaussian_ids'):
            num_gaussians = len(result.gaussian_ids)
            
            # 更多高斯点通常意味着更好的多视角一致性
            consistency_score = min(1.0, num_gaussians / 100.0)  # 100个点为满分
            
            # 考虑空间分布
            if hasattr(result, 'spatial_bounds'):
                bounds = result.spatial_bounds
                volume = (
                    (bounds.get('max_x', 0) - bounds.get('min_x', 0)) *
                    (bounds.get('max_y', 0) - bounds.get('min_y', 0)) *
                    (bounds.get('max_z', 0) - bounds.get('min_z', 0))
                )
                
                # 适中的体积表示良好的空间分布
                if 0.1 <= volume <= 10.0:  # 合理的体积范围
                    consistency_score *= 1.2
                
            return min(1.0, consistency_score)
        
        return 0.5
    
    def _normalize_scores(self, factor_scores: List[Dict]) -> List[Dict]:
        """归一化各因子分数"""
        if not factor_scores:
            return factor_scores
        
        # 提取各因子的分数
        factor_names = ['vector_similarity', 'text_similarity', 'visual_similarity', 
                       'spatial_relevance', 'multi_view_consistency']
        
        # 为每个因子进行归一化
        normalized_scores = []
        
        for scores in factor_scores:
            normalized = scores.copy()
            
            for factor in factor_names:
                # 简单的min-max归一化到[0,1]
                value = scores[factor]
                normalized[factor] = max(0.0, min(1.0, value))
            
            normalized_scores.append(normalized)
        
        return normalized_scores
    
    def _compute_weighted_scores(self, normalized_scores: List[Dict], 
                                intent: QueryIntent) -> List[MultiFactorScore]:
        """计算加权最终分数"""
        weights = intent.priority_weights
        
        final_scores = []
        
        for scores in normalized_scores:
            # 加权求和
            final_score = (
                scores['vector_similarity'] * weights.get('vector_similarity', 0.3) +
                scores['text_similarity'] * weights.get('text_similarity', 0.2) +
                scores['visual_similarity'] * weights.get('visual_similarity', 0.2) +
                scores['spatial_relevance'] * weights.get('spatial_relevance', 0.2) +
                scores['multi_view_consistency'] * weights.get('multi_view_consistency', 0.1)
            )
            
            multi_factor_score = MultiFactorScore(
                cluster_id=scores['cluster_id'],
                vector_similarity=scores['vector_similarity'],
                text_similarity=scores['text_similarity'],
                visual_similarity=scores['visual_similarity'],
                spatial_relevance=scores['spatial_relevance'],
                multi_view_consistency=scores['multi_view_consistency'],
                final_score=final_score
            )
            
            final_scores.append(multi_factor_score)
        
        return final_scores

class IntelligentRAG:
    """智能化RAG系统主流水线"""
    
    def __init__(self, model_path: str, llm_provider: LLMProvider, 
                 vector_db_path: str = None):
        """
        初始化智能RAG系统
        
        Args:
            model_path: 3DGS模型路径
            llm_provider: LLM提供商
            vector_db_path: 向量数据库路径
        """
        self.llm_provider = llm_provider
        
        # 加载3DGS模型信息
        self.model_info = self._load_model_info(model_path)
        
        # 初始化组件
        self.pruner = GaussianPruner(self.model_info)
        
        # 初始化向量数据库（简化版）
        self.vector_index = self._initialize_vector_index(vector_db_path)
        
        # 初始化重排序器（需要CLIP提取器）
        self.clip_extractor = self._initialize_clip_extractor()
        self.reranker = MultiFactorReranker(self.clip_extractor, self.vector_index)
        
        # 缓存系统
        self.intent_cache = {}
        self.result_cache = {}
        
        logger.info("智能化RAG系统初始化完成")
    
    def _load_model_info(self, model_path: str):
        """加载模型信息"""
        # 简化实现：创建模拟的模型信息
        logger.info(f"加载模型: {model_path}")
        
        # 创建模拟的高斯点数据
        num_gaussians = 10000
        means = np.random.normal(0, 2, (num_gaussians, 3))
        
        class ModelInfo:
            def __init__(self, means):
                self.means = means
                self.features = np.random.normal(0, 1, (len(means), 256))  # 模拟特征
        
        return ModelInfo(means)
    
    def _initialize_vector_index(self, vector_db_path: str):
        """初始化向量索引"""
        if vector_db_path and os.path.exists(vector_db_path):
            logger.info(f"加载向量索引: {vector_db_path}")
            index = faiss.read_index(vector_db_path)
        else:
            logger.info("创建新的向量索引")
            # 创建简单的FAISS索引
            dimension = 512  # CLIP特征维度
            index = faiss.IndexFlatIP(dimension)  # 内积索引
        
        return index
    
    def _initialize_clip_extractor(self):
        """初始化CLIP特征提取器"""
        # 简化实现：创建模拟的CLIP提取器
        class MockCLIPExtractor:
            def extract_from_text(self, text: str) -> np.ndarray:
                # 模拟CLIP文本编码
                hash_val = hash(text) % 1000000
                np.random.seed(hash_val)
                return np.random.normal(0, 1, 512)
            
            def extract_from_image(self, image: np.ndarray) -> np.ndarray:
                # 模拟CLIP图像编码
                return np.random.normal(0, 1, 512)
        
        return MockCLIPExtractor()
    
    async def intelligent_search(self, query: str, top_k: int = 10, 
                                downsample_factor: float = 0.3) -> Dict:
        """
        智能搜索主流程
        
        Args:
            query: 自然语言查询
            top_k: 返回的结果数量
            downsample_factor: 降采样因子
            
        Returns:
            搜索结果字典
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"{hash(query)}_{top_k}_{downsample_factor}"
        if cache_key in self.result_cache:
            logger.info("返回缓存结果")
            return self.result_cache[cache_key]
        
        logger.info(f"开始智能搜索: '{query}'")
        
        # 第一阶段：LLM意图分析
        intent = await self._analyze_query_intent(query)
        logger.info(f"查询意图分析完成，类型: {intent.query_type.value}")
        
        # 第二阶段：基于意图的模型剪枝
        pruned_indices = self._prune_gaussians(intent, downsample_factor)
        logger.info(f"模型剪枝完成，保留 {len(pruned_indices)} 个高斯点")
        
        # 第三阶段：初始检索
        initial_results = self._initial_retrieval(query, intent, pruned_indices)
        logger.info(f"初始检索完成，获得 {len(initial_results)} 个候选结果")
        
        # 第四阶段：多因子重排序
        final_results = self._rerank_results(query, intent, initial_results, top_k)
        logger.info(f"重排序完成，返回 {len(final_results)} 个最终结果")
        
        # 构建结果
        search_results = {
            'query': query,
            'intent': intent.to_dict(),
            'processing_time': time.time() - start_time,
            'pruned_gaussians': len(pruned_indices),
            'initial_candidates': len(initial_results),
            'final_results': [score.to_dict() for score in final_results],
            'performance_metrics': self._compute_performance_metrics(
                intent, pruned_indices, initial_results, final_results
            )
        }
        
        # 缓存结果
        self.result_cache[cache_key] = search_results
        
        logger.info(f"智能搜索完成，总耗时: {search_results['processing_time']:.3f}秒")
        return search_results
    
    async def _analyze_query_intent(self, query: str) -> QueryIntent:
        """分析查询意图"""
        # 检查意图缓存
        if query in self.intent_cache:
            return self.intent_cache[query]
        
        # 调用LLM分析意图
        intent = await self.llm_provider.analyze_query_intent(query)
        
        # 缓存意图
        self.intent_cache[query] = intent
        
        return intent
    
    def _prune_gaussians(self, intent: QueryIntent, downsample_factor: float) -> List[int]:
        """基于意图剪枝高斯点"""
        return self.pruner.prune_by_intent(intent, downsample_factor)
    
    def _initial_retrieval(self, query: str, intent: QueryIntent, 
                          gaussian_indices: List[int]) -> List:
        """初始检索"""
        # 提取查询特征
        query_features = self.clip_extractor.extract_from_text(query)
        
        # 创建模拟的检索结果
        results = []
        
        # 简化实现：基于高斯点索引创建聚类结果
        num_clusters = min(50, len(gaussian_indices) // 20)  # 创建聚类
        
        for i in range(num_clusters):
            # 模拟聚类信息
            cluster_gaussians = gaussian_indices[i::num_clusters]
            
            if not cluster_gaussians:
                continue
            
            # 计算聚类中心
            cluster_positions = self.model_info.means[cluster_gaussians]
            center_position = np.mean(cluster_positions, axis=0)
            
            # 计算边界
            min_bounds = np.min(cluster_positions, axis=0)
            max_bounds = np.max(cluster_positions, axis=0)
            
            # 模拟语义标签
            semantic_labels = self._generate_semantic_labels(center_position)
            
            # 计算相似度分数
            cluster_features = np.mean(self.model_info.features[cluster_gaussians], axis=0)
            similarity_score = np.dot(query_features, cluster_features) / (
                np.linalg.norm(query_features) * np.linalg.norm(cluster_features)
            )
            
            # 创建结果对象
            result = type('SearchResult', (), {
                'cluster_id': i,
                'gaussian_ids': cluster_gaussians,
                'center_position': center_position,
                'spatial_bounds': {
                    'min_x': float(min_bounds[0]), 'max_x': float(max_bounds[0]),
                    'min_y': float(min_bounds[1]), 'max_y': float(max_bounds[1]),
                    'min_z': float(min_bounds[2]), 'max_z': float(max_bounds[2])
                },
                'semantic_labels': semantic_labels,
                'similarity_score': float(similarity_score),
                'description': f"Cluster {i} with {len(cluster_gaussians)} gaussians"
            })()
            
            results.append(result)
        
        # 按相似度排序
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:20]  # 返回前20个候选结果
    
    def _generate_semantic_labels(self, position: np.ndarray) -> List[str]:
        """基于位置生成语义标签"""
        x, y, z = position
        labels = []
        
        # 基于高度
        if y < 0.1:
            labels.append('floor')
        elif y > 2.0:
            labels.append('ceiling')
        elif 0.5 < y < 1.5:
            labels.extend(['furniture', 'table', 'chair'])
        
        # 基于水平位置
        if abs(x) > 2 or abs(z) > 2:
            labels.extend(['wall', 'boundary'])
        else:
            labels.extend(['interior', 'object'])
        
        # 添加一些随机的语义标签
        possible_labels = ['wooden', 'modern', 'comfortable', 'decorative', 'functional']
        labels.extend(np.random.choice(possible_labels, 2, replace=False))
        
        return labels
    
    def _rerank_results(self, query: str, intent: QueryIntent, 
                       initial_results: List, top_k: int) -> List[MultiFactorScore]:
        """重排序结果"""
        return self.reranker.rerank_results(query, intent, initial_results, top_k)
    
    def _compute_performance_metrics(self, intent: QueryIntent, pruned_indices: List[int],
                                   initial_results: List, final_results: List) -> Dict:
        """计算性能指标"""
        total_gaussians = len(self.model_info.means)
        pruning_ratio = len(pruned_indices) / total_gaussians
        
        return {
            'total_gaussians': total_gaussians,
            'pruned_gaussians': len(pruned_indices),
            'pruning_ratio': pruning_ratio,
            'initial_candidates': len(initial_results),
            'final_results': len(final_results),
            'intent_confidence': intent.confidence,
            'query_type': intent.query_type.value,
            'speedup_estimate': 1.0 / pruning_ratio if pruning_ratio > 0 else 1.0
        }
    
    def get_system_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            'model_gaussians': len(self.model_info.means),
            'cached_intents': len(self.intent_cache),
            'cached_results': len(self.result_cache),
            'vector_index_size': self.vector_index.ntotal if hasattr(self.vector_index, 'ntotal') else 0
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.intent_cache.clear()
        self.result_cache.clear()
        logger.info("缓存已清空")

# 工厂函数和便捷接口
def create_intelligent_rag(model_path: str, api_key: str, 
                          base_url: str = "https://api.openai.com/v1", 
                          model: str = "gpt-4", **kwargs) -> IntelligentRAG:
    """
    创建智能RAG系统
    
    Args:
        model_path: 3DGS模型路径
        api_key: OpenAI兼容API密钥
        base_url: API基础URL (默认OpenAI, 也支持其他兼容提供商)
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        配置好的IntelligentRAG实例
    """
    # 创建OpenAI兼容的LLM提供商
    if not api_key:
        raise ValueError("需要提供API密钥")
    
    # 创建OpenAI客户端，支持自定义base_url以兼容其他提供商
    llm_provider = OpenAICompatibleProvider(
        api_key=api_key, 
        base_url=base_url,
        model=model
    )
    
    # 创建智能RAG系统
    return IntelligentRAG(
        model_path=model_path,
        llm_provider=llm_provider,
        vector_db_path=kwargs.get('vector_db_path')
    )

async def quick_search(query: str, model_path: str, api_key: str,
                      base_url: str = "https://api.openai.com/v1",
                      model: str = "gpt-4", top_k: int = 10) -> Dict:
    """
    快速搜索便捷函数
    
    Args:
        query: 搜索查询
        model_path: 模型路径
        api_key: OpenAI兼容API密钥
        base_url: API基础URL
        model: 模型名称
        top_k: 返回结果数量
        
    Returns:
        搜索结果
    """
    rag_system = create_intelligent_rag(
        model_path=model_path, 
        api_key=api_key, 
        base_url=base_url,
        model=model
    )
    return await rag_system.intelligent_search(query, top_k)

if __name__ == "__main__":
    import asyncio
    import os
    
    async def demo():
        """演示智能RAG系统"""
        # 从环境变量获取API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("请设置OPENAI_API_KEY环境变量")
            return
        
        # 创建系统
        rag = create_intelligent_rag(
            model_path="./model/scene.ply",
            api_key=api_key,
            model="gpt-4"
        )
        
        # 测试查询
        queries = [
            "找一把红色的椅子",
            "客厅中心的桌子",
            "靠近窗户的家具",
            "现代风格的装饰品"
        ]
        
        for query in queries:
            print(f"\n搜索查询: {query}")
            print("-" * 50)
            
            results = await rag.intelligent_search(query, top_k=5)
            
            print(f"处理时间: {results['processing_time']:.3f}秒")
            print(f"剪枝效果: {results['pruned_gaussians']}/{results['performance_metrics']['total_gaussians']}")
            print(f"查询类型: {results['intent']['query_type']}")
            
            print("\n前3个结果:")
            for i, result in enumerate(results['final_results'][:3]):
                print(f"  {i+1}. 聚类{result['cluster_id']} - 得分: {result['final_score']:.3f}")
                print(f"     向量相似度: {result['vector_similarity']:.3f}")
                print(f"     文本相似度: {result['text_similarity']:.3f}")
                print(f"     空间相关性: {result['spatial_relevance']:.3f}")
        
        # 显示系统统计
        print(f"\n系统统计: {rag.get_system_stats()}")
    
    # 运行演示
    asyncio.run(demo())