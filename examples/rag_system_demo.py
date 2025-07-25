#!/usr/bin/env python3
"""
GSRender RAG系统使用示例
展示如何构建和使用基于CLIP和FAISS的3D高斯点RAG后端
"""

import numpy as np
import cv2
import os
import sys
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# 添加Python模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

# 导入必要模块
try:
    import gsrender
    print("✓ GSRender模块导入成功")
except ImportError as e:
    print(f"✗ GSRender模块导入失败: {e}")
    print("请先编译Python扩展: python setup.py build_ext --inplace")
    sys.exit(1)

try:
    from gsrender_rag import GSRenderRAGBackend, QueryResult
    print("✓ GSRender RAG模块导入成功")
except ImportError as e:
    print(f"✗ GSRender RAG模块导入失败: {e}")
    print("请检查python/gsrender_rag.py文件")
    sys.exit(1)

def create_multi_view_camera_poses(num_views: int = 12, radius: float = 5.0, 
                                  height_variation: float = 2.0) -> List[np.ndarray]:
    """
    创建多视角相机位姿
    
    Args:
        num_views: 视角数量
        radius: 相机距离中心的半径
        height_variation: 高度变化范围
        
    Returns:
        相机位姿列表
    """
    camera_poses = []
    
    for i in range(num_views):
        # 水平角度
        angle = i * 2 * np.pi / num_views
        
        # 高度变化（形成椭圆形轨迹）
        height = height_variation * np.sin(2 * angle)
        
        # 相机位置
        cam_x = radius * np.cos(angle)
        cam_z = radius * np.sin(angle)
        cam_y = height
        
        # 创建朝向原点的视图矩阵
        # 相机坐标系：+X右，+Y上，+Z后（朝向场景）
        forward = np.array([0, 0, 0]) - np.array([cam_x, cam_y, cam_z])  # 朝向原点
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 1, 0])  # 世界坐标上方向
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # 构建4x4变换矩阵（相机到世界坐标）
        view_matrix = np.array([
            [right[0], up[0], -forward[0], cam_x],
            [right[1], up[1], -forward[1], cam_y],
            [right[2], up[2], -forward[2], cam_z],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        camera_poses.append(view_matrix)
    
    return camera_poses

def setup_rag_system(ply_path: str, num_views: int = 12) -> GSRenderRAGBackend:
    """
    设置和初始化RAG系统
    
    Args:
        ply_path: PLY模型文件路径
        num_views: 用于构建RAG的视角数量
        
    Returns:
        初始化的RAG后端
    """
    print("\\n=== 设置RAG系统 ===")
    
    # 检查模型文件
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"模型文件不存在: {ply_path}")
    
    # 创建GSRender接口
    print("初始化GSRender接口...")
    renderer = gsrender.GSRenderInterface()
    device = "cuda" if gsrender.torch.cuda.is_available() else "cpu"
    
    success = renderer.load_model(ply_path, device)
    if not success:
        raise RuntimeError("模型加载失败")
    
    model_info = renderer.get_model_info()
    print(f"✓ 模型加载成功: {model_info.num_gaussians} 个高斯点")
    
    # 创建RAG后端
    print("初始化RAG后端...")
    rag_backend = GSRenderRAGBackend(device=device)
    
    # 创建多视角相机位姿
    print(f"创建 {num_views} 个视角的相机位姿...")
    camera_poses = create_multi_view_camera_poses(num_views)
    
    # 构建RAG系统
    print("构建RAG系统（这可能需要几分钟）...")
    clustering_params = {
        'semantic_threshold': 0.6,  # 语义相似度阈值
        'spatial_threshold': 1.5,   # 空间距离阈值（米）
        'min_cluster_size': 3       # 最小聚类大小
    }
    
    rag_backend.build_rag_from_renders(
        renderer, camera_poses, **clustering_params
    )
    
    print("✓ RAG系统构建完成")
    return rag_backend

def demonstrate_text_queries(rag_backend: GSRenderRAGBackend):
    """演示文本查询功能"""
    print("\\n=== 文本查询演示 ===")
    
    # 预定义查询
    queries = [
        "furniture in the room",
        "table and chairs", 
        "lighting fixtures",
        "decorative objects",
        "electronic devices",
        "books and reading materials",
        "plants and nature",
        "wall decorations"
    ]
    
    query_results = {}
    
    for query in queries:
        print(f"\\n查询: '{query}'")
        results = rag_backend.query(query, top_k=3)
        
        query_results[query] = results
        
        if results:
            print(f"找到 {len(results)} 个匹配结果:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.description}")
                print(f"     相似度: {result.similarity_score:.3f}")
                print(f"     语义标签: {', '.join(result.semantic_labels[:2])}")
                print(f"     高斯点数: {len(result.gaussian_ids)}")
                
                # 空间信息
                bounds = result.spatial_bounds
                if bounds:
                    center_x = (bounds['min_x'] + bounds['max_x']) / 2
                    center_y = (bounds['min_y'] + bounds['max_y']) / 2
                    center_z = (bounds['min_z'] + bounds['max_z']) / 2
                    print(f"     空间中心: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")
        else:
            print("  未找到匹配结果")
    
    return query_results

def demonstrate_image_queries(rag_backend: GSRenderRAGBackend, renderer):
    """演示图像查询功能"""
    print("\\n=== 图像查询演示 ===")
    
    # 渲染一个参考图像
    reference_pose = np.array([
        [1.0, 0.0, 0.0, 3.0],
        [0.0, 0.8, -0.6, 2.0],
        [0.0, 0.6, 0.8, 4.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    camera_params = gsrender.create_camera_params(
        view_matrix=reference_pose,
        width=512, height=512,
        fx=256, fy=256
    )
    
    reference_result = renderer.render(camera_params)
    reference_image = reference_result.rgb_image
    
    # 保存参考图像
    cv2.imwrite("reference_query_image.png", 
                cv2.cvtColor((reference_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("✓ 参考查询图像已保存: reference_query_image.png")
    
    # 使用图像查询
    print("执行图像查询...")
    results = rag_backend.query_by_image(reference_image, top_k=5)
    
    if results:
        print(f"图像查询找到 {len(results)} 个匹配结果:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.description}")
            print(f"     相似度: {result.similarity_score:.3f}")
    else:
        print("图像查询未找到匹配结果")
    
    return results

def analyze_clustering_results(rag_backend: GSRenderRAGBackend):
    """分析聚类结果"""
    print("\\n=== 聚类结果分析 ===")
    
    # 获取统计信息
    stats = rag_backend.get_cluster_statistics()
    
    print("系统统计信息:")
    print(f"  聚类总数: {stats['total_clusters']}")
    print(f"  高斯点总数: {stats['total_gaussians']}")
    print(f"  平均每聚类高斯点数: {stats['avg_gaussians_per_cluster']:.2f}")
    print(f"  特征维度: {stats['feature_dimension']}")
    
    # 语义标签分布
    print("\\n语义标签分布:")
    label_counts = stats['semantic_label_counts']
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    for label, count in sorted_labels[:10]:  # 显示前10个
        print(f"  {label}: {count}")
    
    # 创建可视化图表
    create_analysis_charts(stats)
    
    return stats

def create_analysis_charts(stats: Dict):
    """创建分析图表"""
    print("\\n创建分析图表...")
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 语义标签分布饼图
    label_counts = stats['semantic_label_counts']
    if label_counts:
        top_labels = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:8])
        others_count = sum(label_counts.values()) - sum(top_labels.values())
        if others_count > 0:
            top_labels['others'] = others_count
        
        ax1.pie(top_labels.values(), labels=top_labels.keys(), autopct='%1.1f%%')
        ax1.set_title('Semantic Label Distribution')
    
    # 2. 系统概览条形图
    overview_data = {
        'Total Clusters': stats['total_clusters'],
        'Total Gaussians': stats['total_gaussians'],
        'Avg Gaussians/Cluster': int(stats['avg_gaussians_per_cluster'])
    }
    
    ax2.bar(overview_data.keys(), overview_data.values())
    ax2.set_title('System Overview')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 聚类大小分布（模拟数据）
    # 在实际应用中，这里应该从实际聚类数据中获取
    cluster_sizes = np.random.gamma(5, 2, stats['total_clusters'])  # 模拟分布
    ax3.hist(cluster_sizes, bins=20, alpha=0.7)
    ax3.set_title('Cluster Size Distribution')
    ax3.set_xlabel('Gaussians per Cluster')
    ax3.set_ylabel('Frequency')
    
    # 4. 特征维度信息
    feature_info = {
        'Feature Dimension': stats['feature_dimension'],
        'Index Size (approx)': stats['total_clusters'] * stats['feature_dimension'] * 4  # bytes
    }
    
    ax4.bar(['Feature Dim', 'Index Size (KB)'], 
           [feature_info['Feature Dimension'], feature_info['Index Size'] / 1024])
    ax4.set_title('Feature Information')
    
    plt.tight_layout()
    plt.savefig('rag_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 分析图表已保存: rag_analysis.png")

def demonstrate_persistent_storage(rag_backend: GSRenderRAGBackend):
    """演示持久化存储功能"""
    print("\\n=== 持久化存储演示 ===")
    
    # 保存RAG系统
    save_path = "scene_rag_index"
    print(f"保存RAG系统到 {save_path}...")
    rag_backend.save(save_path)
    
    # 创建新的RAG后端并加载
    print("创建新的RAG后端并加载...")
    new_rag_backend = GSRenderRAGBackend()
    new_rag_backend.load(save_path)
    
    # 测试加载的系统
    print("测试加载的系统...")
    test_query = "furniture"
    results = new_rag_backend.query(test_query, top_k=2)
    
    if results:
        print(f"✓ 加载的系统正常工作，查询 '{test_query}' 找到 {len(results)} 个结果")
        for result in results:
            print(f"  - {result.description}")
    else:
        print("✗ 加载的系统查询无结果")
    
    return new_rag_backend

def create_query_interface():
    """创建交互式查询接口"""
    print("\\n=== 交互式查询接口 ===")
    print("输入文本查询，输入 'quit' 退出")
    
    # 加载已保存的RAG系统
    save_path = "scene_rag_index"
    if os.path.exists(f"{save_path}.faiss"):
        rag_backend = GSRenderRAGBackend()
        rag_backend.load(save_path)
        print("✓ RAG系统加载成功")
        
        while True:
            query = input("\\n查询> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                try:
                    results = rag_backend.query(query, top_k=3)
                    
                    if results:
                        print(f"\\n找到 {len(results)} 个结果:")
                        for i, result in enumerate(results, 1):
                            print(f"{i}. {result.description}")
                            print(f"   相似度: {result.similarity_score:.3f}")
                    else:
                        print("未找到匹配结果")
                        
                except Exception as e:
                    print(f"查询出错: {e}")
    else:
        print("✗ 未找到保存的RAG系统，请先运行完整示例")

def export_results(query_results: Dict, stats: Dict):
    """导出结果到JSON"""
    print("\\n=== 导出结果 ===")
    
    # 准备导出数据
    export_data = {
        'system_statistics': stats,
        'query_results': {}
    }
    
    # 转换查询结果
    for query, results in query_results.items():
        export_data['query_results'][query] = []
        for result in results:
            result_dict = {
                'cluster_id': result.cluster_id,
                'similarity_score': result.similarity_score,
                'description': result.description,
                'semantic_labels': result.semantic_labels,
                'confidence_scores': result.confidence_scores,
                'gaussian_count': len(result.gaussian_ids),
                'spatial_bounds': result.spatial_bounds
            }
            export_data['query_results'][query].append(result_dict)
    
    # 保存到JSON文件
    with open('rag_results.json', 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    print("✓ 结果已导出到 rag_results.json")

def main():
    """主函数"""
    print("GSRender RAG系统完整示例")
    print("=" * 60)
    
    # 配置参数
    ply_path = "../model/model.ply"  # 请替换为实际路径
    num_views = 8  # 减少视角数量以加快演示速度
    
    try:
        # 1. 设置RAG系统
        rag_backend = setup_rag_system(ply_path, num_views)
        
        # 2. 演示文本查询
        query_results = demonstrate_text_queries(rag_backend)
        
        # 3. 分析聚类结果
        stats = analyze_clustering_results(rag_backend)
        
        # 4. 演示持久化存储
        demonstrate_persistent_storage(rag_backend)
        
        # 5. 导出结果
        export_results(query_results, stats)
        
        print("\\n" + "=" * 60)
        print("✓ 所有RAG系统演示完成!")
        print("\\n生成的文件:")
        print("  - scene_rag_index.faiss: FAISS向量索引")
        print("  - scene_rag_index.pkl: 聚类元数据")
        print("  - rag_analysis.png: 分析图表")
        print("  - rag_results.json: 查询结果")
        print("  - reference_query_image.png: 参考查询图像")
        
        # 6. 可选的交互式查询
        interactive = input("\\n是否启动交互式查询? (y/N): ").strip().lower()
        if interactive in ['y', 'yes']:
            create_query_interface()
        
    except FileNotFoundError as e:
        print(f"\\n✗ 文件未找到: {e}")
        print("请确保模型文件存在，或修改ply_path变量")
    except Exception as e:
        print(f"\\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()