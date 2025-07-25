#!/usr/bin/env python3
"""
GSRender RAG系统优化示例
展示高级功能：增量更新、多模态查询、聚类优化等
"""

import numpy as np
import torch
import cv2
import os
import sys
import json
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass

# 添加Python模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import gsrender
    from gsrender_rag import GSRenderRAGBackend, GaussianCluster, QueryResult
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

@dataclass
class BenchmarkResult:
    """性能测试结果"""
    operation: str
    duration: float
    details: Dict

class AdvancedRAGSystem(GSRenderRAGBackend):
    """高级RAG系统，支持增量更新和优化功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_cache = {}
        self.update_history = []
    
    def incremental_update(self, 
                          new_render_results: List[Dict],
                          new_camera_poses: List[np.ndarray],
                          gaussian_positions: np.ndarray,
                          merge_threshold: float = 0.8):
        """
        增量更新RAG系统
        
        Args:
            new_render_results: 新的渲染结果
            new_camera_poses: 新的相机位姿
            gaussian_positions: 高斯点位置
            merge_threshold: 聚类合并阈值
        """
        print("执行增量更新...")
        start_time = time.time()
        
        if not self.is_built:
            raise RuntimeError("系统未构建，请先调用build_rag_from_renders()")
        
        # 分析新的渲染结果
        new_clusters = self.cluster_analyzer.analyze_gaussians_from_renders(
            new_render_results, new_camera_poses, gaussian_positions,
            semantic_threshold=0.6, spatial_threshold=1.5, min_cluster_size=3
        )
        
        # 合并到现有聚类
        existing_clusters = list(self.vector_index.cluster_metadata.values())
        merged_clusters = self._merge_clusters(existing_clusters, new_clusters, merge_threshold)
        
        # 重建索引
        self.vector_index.build_index(merged_clusters)
        
        duration = time.time() - start_time
        self.update_history.append({
            'timestamp': time.time(),
            'new_clusters': len(new_clusters),
            'merged_clusters': len(merged_clusters),
            'duration': duration
        })
        
        print(f"✓ 增量更新完成，用时 {duration:.2f}s")
        return merged_clusters
    
    def _merge_clusters(self, 
                       existing_clusters: List[GaussianCluster],
                       new_clusters: List[GaussianCluster],
                       threshold: float) -> List[GaussianCluster]:
        """合并聚类"""
        print(f"合并 {len(existing_clusters)} 个现有聚类和 {len(new_clusters)} 个新聚类...")
        
        merged_clusters = existing_clusters.copy()
        
        for new_cluster in new_clusters:
            best_match_idx = -1
            best_similarity = -1
            
            # 寻找最相似的现有聚类
            for i, existing_cluster in enumerate(merged_clusters):
                # 计算特征相似度
                feature_sim = np.dot(new_cluster.center_feature, existing_cluster.center_feature)
                
                # 计算空间距离
                spatial_dist = np.linalg.norm(new_cluster.center_position - existing_cluster.center_position)
                
                # 综合相似度（特征权重0.7，空间权重0.3）
                combined_sim = 0.7 * feature_sim - 0.3 * (spatial_dist / 10.0)  # 归一化空间距离
                
                if combined_sim > best_similarity:
                    best_similarity = combined_sim
                    best_match_idx = i
            
            # 如果相似度足够高，合并聚类
            if best_similarity > threshold and best_match_idx != -1:
                existing_cluster = merged_clusters[best_match_idx]
                
                # 合并高斯点
                combined_gaussian_ids = list(set(existing_cluster.gaussian_ids + new_cluster.gaussian_ids))
                combined_positions = np.vstack([existing_cluster.positions, new_cluster.positions])
                combined_features = np.vstack([existing_cluster.features, new_cluster.features])
                
                # 更新聚类
                merged_clusters[best_match_idx] = GaussianCluster(
                    cluster_id=existing_cluster.cluster_id,
                    gaussian_ids=combined_gaussian_ids,
                    positions=combined_positions,
                    features=combined_features,
                    semantic_labels=existing_cluster.semantic_labels,  # 保持原有标签
                    confidence_scores=existing_cluster.confidence_scores,
                    center_position=np.mean(combined_positions, axis=0),
                    center_feature=np.mean(combined_features, axis=0),
                    spatial_bounds=self.cluster_analyzer._compute_spatial_bounds(combined_positions)
                )
            else:
                # 添加为新聚类
                new_cluster.cluster_id = len(merged_clusters)
                merged_clusters.append(new_cluster)
        
        print(f"合并后共有 {len(merged_clusters)} 个聚类")
        return merged_clusters
    
    def multimodal_query(self, 
                        text: Optional[str] = None,
                        image: Optional[np.ndarray] = None,
                        spatial_bounds: Optional[Dict] = None,
                        semantic_filter: Optional[List[str]] = None,
                        top_k: int = 5) -> List[QueryResult]:
        """
        多模态查询
        
        Args:
            text: 文本查询
            image: 图像查询
            spatial_bounds: 空间边界过滤 {min_x, max_x, min_y, max_y, min_z, max_z}
            semantic_filter: 语义标签过滤
            top_k: 返回结果数量
        """
        if not self.is_built:
            raise RuntimeError("系统未构建")
        
        print("执行多模态查询...")
        
        # 提取查询特征
        query_features = []
        
        if text:
            text_feature = self.clip_extractor.extract_from_text(text)
            query_features.append(('text', text_feature))
        
        if image is not None:
            image_feature = self.clip_extractor.extract_from_image(image)
            query_features.append(('image', image_feature))
        
        if not query_features:
            raise ValueError("至少需要提供文本或图像查询")
        
        # 合并多个查询特征（加权平均）
        if len(query_features) == 1:
            combined_feature = query_features[0][1]
        else:
            # 文本权重0.6，图像权重0.4
            weights = {'text': 0.6, 'image': 0.4}
            combined_feature = np.zeros_like(query_features[0][1])
            total_weight = 0
            
            for modality, feature in query_features:
                weight = weights.get(modality, 1.0)
                combined_feature += weight * feature
                total_weight += weight
            
            combined_feature /= total_weight
        
        # 搜索
        results = self.vector_index.search(combined_feature, top_k * 2)  # 获取更多结果用于过滤
        
        # 应用过滤器
        filtered_results = []
        for result in results:
            # 空间过滤
            if spatial_bounds:
                cluster_bounds = result.spatial_bounds
                if not self._check_spatial_overlap(cluster_bounds, spatial_bounds):
                    continue
            
            # 语义过滤
            if semantic_filter:
                if not any(label in result.semantic_labels for label in semantic_filter):
                    continue
            
            filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break
        
        print(f"多模态查询找到 {len(filtered_results)} 个结果")
        return filtered_results
    
    def _check_spatial_overlap(self, cluster_bounds: Dict, query_bounds: Dict) -> bool:
        """检查空间边界重叠"""
        return (cluster_bounds.get('max_x', float('-inf')) >= query_bounds.get('min_x', float('-inf')) and
                cluster_bounds.get('min_x', float('inf')) <= query_bounds.get('max_x', float('inf')) and
                cluster_bounds.get('max_y', float('-inf')) >= query_bounds.get('min_y', float('-inf')) and
                cluster_bounds.get('min_y', float('inf')) <= query_bounds.get('max_y', float('inf')) and
                cluster_bounds.get('max_z', float('-inf')) >= query_bounds.get('min_z', float('-inf')) and
                cluster_bounds.get('min_z', float('inf')) <= query_bounds.get('max_z', float('inf')))
    
    def benchmark_performance(self, test_queries: List[str], num_runs: int = 10) -> List[BenchmarkResult]:
        """性能基准测试"""
        print(f"执行性能基准测试，{len(test_queries)} 个查询，每个运行 {num_runs} 次...")
        
        results = []
        
        for query in test_queries:
            print(f"测试查询: '{query}'")
            
            # 预热
            self.query(query, top_k=5)
            
            # 测试多次运行
            durations = []
            for _ in range(num_runs):
                start_time = time.time()
                query_results = self.query(query, top_k=5)
                duration = time.time() - start_time
                durations.append(duration)
            
            # 统计结果
            benchmark_result = BenchmarkResult(
                operation=f"query_{query[:20]}",
                duration=np.mean(durations),
                details={
                    'query': query,
                    'runs': num_runs,
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'std_duration': np.std(durations),
                    'result_count': len(query_results) if query_results else 0
                }
            )
            
            results.append(benchmark_result)
            print(f"  平均用时: {benchmark_result.duration:.4f}s")
        
        return results
    
    def optimize_index(self, target_recall: float = 0.95):
        """优化FAISS索引参数"""
        print(f"优化索引以达到 {target_recall} 召回率...")
        
        if not self.is_built:
            raise RuntimeError("系统未构建")
        
        # 这里可以实现索引参数调优
        # 例如：调整IVF的nprobe参数、HNSW的ef参数等
        
        if hasattr(self.vector_index.index, 'nprobe'):
            # IVF索引优化
            original_nprobe = self.vector_index.index.nprobe
            
            # 测试不同的nprobe值
            best_nprobe = original_nprobe
            best_recall = 0
            
            for nprobe in [1, 2, 4, 8, 16, 32]:
                if nprobe > self.vector_index.index.nlist:
                    break
                
                self.vector_index.index.nprobe = nprobe
                recall = self._measure_recall()
                
                print(f"  nprobe={nprobe}, recall={recall:.3f}")
                
                if recall >= target_recall and recall > best_recall:
                    best_recall = recall
                    best_nprobe = nprobe
            
            self.vector_index.index.nprobe = best_nprobe
            print(f"✓ 索引优化完成，使用 nprobe={best_nprobe}, recall={best_recall:.3f}")
        
        elif hasattr(self.vector_index.index, 'hnsw'):
            # HNSW索引优化
            print("HNSW索引优化暂未实现")
        
        else:
            print("当前索引类型不支持参数优化")
    
    def _measure_recall(self, num_test_queries: int = 50) -> float:
        """测量索引召回率"""
        if not self.cluster_metadata:
            return 0.0
        
        # 随机选择一些聚类作为查询
        clusters = list(self.vector_index.cluster_metadata.values())
        test_clusters = np.random.choice(clusters, min(num_test_queries, len(clusters)), replace=False)
        
        total_recall = 0
        
        for cluster in test_clusters:
            # 使用聚类的特征作为查询
            results = self.vector_index.search(cluster.center_feature, k=10)
            
            # 检查原始聚类是否在结果中
            found = any(result.cluster_id == cluster.cluster_id for result in results)
            total_recall += int(found)
        
        return total_recall / len(test_clusters)

def demonstrate_incremental_update():
    """演示增量更新功能"""
    print("\\n=== 增量更新演示 ===")
    
    # 这里需要根据实际场景实现
    # 模拟新的渲染数据
    print("增量更新功能需要新的渲染数据，此处为演示框架")

def demonstrate_multimodal_queries(rag_system: AdvancedRAGSystem):
    """演示多模态查询"""
    print("\\n=== 多模态查询演示 ===")
    
    # 1. 纯文本查询
    print("1. 纯文本查询")
    results = rag_system.multimodal_query(text="modern furniture", top_k=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.description} (相似度: {result.similarity_score:.3f})")
    
    # 2. 带空间过滤的查询
    print("\\n2. 带空间过滤的查询")
    spatial_bounds = {'min_x': -2, 'max_x': 2, 'min_y': -1, 'max_y': 3, 'min_z': -2, 'max_z': 2}
    results = rag_system.multimodal_query(
        text="furniture", 
        spatial_bounds=spatial_bounds,
        top_k=3
    )
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.description} (空间受限)")
    
    # 3. 带语义过滤的查询
    print("\\n3. 带语义过滤的查询")
    results = rag_system.multimodal_query(
        text="room objects",
        semantic_filter=['table', 'chair', 'furniture'],
        top_k=3
    )
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.description} (语义过滤)")

def run_performance_benchmark(rag_system: AdvancedRAGSystem):
    """运行性能基准测试"""
    print("\\n=== 性能基准测试 ===")
    
    test_queries = [
        "furniture",
        "modern design",
        "lighting",
        "decorative objects",
        "electronic devices"
    ]
    
    benchmark_results = rag_system.benchmark_performance(test_queries, num_runs=5)
    
    print("\\n基准测试结果:")
    print("-" * 60)
    for result in benchmark_results:
        print(f"查询: {result.details['query'][:30]}")
        print(f"  平均用时: {result.duration:.4f}s")
        print(f"  最小用时: {result.details['min_duration']:.4f}s")
        print(f"  最大用时: {result.details['max_duration']:.4f}s")
        print(f"  标准差: {result.details['std_duration']:.4f}s")
        print(f"  结果数: {result.details['result_count']}")
        print()
    
    # 保存基准测试结果
    benchmark_data = {
        'results': [
            {
                'query': r.details['query'],
                'avg_duration': r.duration,
                'min_duration': r.details['min_duration'],
                'max_duration': r.details['max_duration'],
                'std_duration': r.details['std_duration'],
                'result_count': r.details['result_count']
            }
            for r in benchmark_results
        ],
        'system_info': rag_system.get_cluster_statistics()
    }
    
    with open('performance_benchmark.json', 'w') as f:
        json.dump(benchmark_data, f, indent=2, default=str)
    
    print("✓ 基准测试结果已保存到 performance_benchmark.json")
    
    return benchmark_results

def demonstrate_index_optimization(rag_system: AdvancedRAGSystem):
    """演示索引优化"""
    print("\\n=== 索引优化演示 ===")
    
    try:
        rag_system.optimize_index(target_recall=0.95)
    except Exception as e:
        print(f"索引优化出错: {e}")
        print("可能是索引类型不支持优化或数据量太小")

def create_advanced_demo():
    """创建高级功能演示"""
    print("GSRender RAG系统高级功能演示")
    print("=" * 60)
    
    # 检查是否有已保存的系统
    save_path = "scene_rag_index"
    if not os.path.exists(f"{save_path}.faiss"):
        print("✗ 未找到已保存的RAG系统")
        print("请先运行 rag_system_demo.py 构建基础系统")
        return
    
    try:
        # 加载高级RAG系统
        rag_system = AdvancedRAGSystem()
        rag_system.load(save_path)
        print("✓ 高级RAG系统加载成功")
        
        # 演示功能
        demonstrate_multimodal_queries(rag_system)
        benchmark_results = run_performance_benchmark(rag_system)
        demonstrate_index_optimization(rag_system)
        
        print("\\n" + "=" * 60)
        print("✓ 高级功能演示完成!")
        print("\\n生成的文件:")
        print("  - performance_benchmark.json: 性能基准测试结果")
        
    except Exception as e:
        print(f"\\n✗ 演示出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_advanced_demo()