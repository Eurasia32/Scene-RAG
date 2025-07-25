#!/usr/bin/env python3
"""
动态数据库演示 - 智能化RAG系统的实际应用示例
展示如何通过LLM意图分析实现动态数据库式的高效3D场景检索
"""

import asyncio
import sys
import os
import time
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

# 添加python目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from intelligent_rag import (
    create_intelligent_rag, 
    QueryType, 
    quick_search,
    IntelligentRAG,
    LocalLLMProvider,
    OpenAIProvider
)

class DynamicDatabaseDemo:
    """动态数据库演示类"""
    
    def __init__(self, model_path: str = "./model/scene.ply"):
        """
        初始化演示
        
        Args:
            model_path: 3DGS模型路径
        """
        self.model_path = model_path
        self.rag_system = None
        self.demo_queries = [
            {
                "query": "找一把红色的木质椅子",
                "expected_type": QueryType.OBJECT_SEARCH,
                "description": "对象搜索：具体物体+视觉属性"
            },
            {
                "query": "客厅中心附近的家具",
                "expected_type": QueryType.SPATIAL_QUERY,
                "description": "空间查询：位置约束+对象类别"
            },
            {
                "query": "现代风格的装饰品，最好是金属材质",
                "expected_type": QueryType.VISUAL_SEARCH,
                "description": "视觉搜索：风格+材质属性"
            },
            {
                "query": "客厅里靠近窗户的舒适座椅",
                "expected_type": QueryType.COMPOSITE_QUERY,
                "description": "复合查询：场景+位置+功能+属性"
            },
            {
                "query": "书房的办公设备",
                "expected_type": QueryType.SEMANTIC_QUERY,
                "description": "语义查询：场景上下文+功能分类"
            }
        ]
    
    async def initialize_system(self, provider_type: str = "local", api_key: str = None):
        """初始化RAG系统"""
        print("🚀 初始化智能RAG系统...")
        start_time = time.time()
        
        self.rag_system = create_intelligent_rag(
            model_path=self.model_path,
            provider_type=provider_type,
            api_key=api_key
        )
        
        init_time = time.time() - start_time
        stats = self.rag_system.get_system_stats()
        
        print(f"✅ 系统初始化完成 ({init_time:.3f}秒)")
        print(f"📊 模型信息: {stats['model_gaussians']:,} 个高斯点")
        print(f"💾 向量索引: {stats['vector_index_size']} 个向量")
        print("-" * 60)
    
    async def demonstrate_intent_analysis(self):
        """演示查询意图分析"""
        print("🧠 LLM查询意图分析演示")
        print("=" * 60)
        
        for i, demo in enumerate(self.demo_queries, 1):
            print(f"\n{i}. 查询: \"{demo['query']}\"")
            print(f"   类型: {demo['description']}")
            
            # 分析意图
            intent = await self.rag_system._analyze_query_intent(demo['query'])
            
            print(f"   🎯 解析结果:")
            print(f"      查询类型: {intent.query_type.value}")
            print(f"      主要对象: {intent.primary_objects}")
            print(f"      空间约束: {intent.spatial_constraints}")
            print(f"      视觉属性: {intent.visual_attributes}")
            print(f"      置信度: {intent.confidence:.3f}")
            print(f"      权重分配: {intent.priority_weights}")
            
            # 验证类型匹配
            if intent.query_type == demo['expected_type']:
                print("      ✅ 意图识别正确")
            else:
                print(f"      ⚠️  期望类型: {demo['expected_type'].value}")
        
        print("\n" + "=" * 60)
    
    async def demonstrate_dynamic_pruning(self):
        """演示动态剪枝效果"""
        print("✂️ 动态模型剪枝演示")
        print("=" * 60)
        
        # 测试不同的降采样因子
        downsample_factors = [0.1, 0.3, 0.5, 0.8]
        test_query = "客厅中心的桌子"
        
        print(f"测试查询: \"{test_query}\"")
        print(f"原始模型: {self.rag_system.get_system_stats()['model_gaussians']:,} 个高斯点")
        print()
        
        intent = await self.rag_system._analyze_query_intent(test_query)
        
        pruning_results = []
        
        for factor in downsample_factors:
            start_time = time.time()
            
            # 执行剪枝
            pruned_indices = self.rag_system._prune_gaussians(intent, factor)
            
            prune_time = time.time() - start_time
            original_count = len(self.rag_system.model_info.means)
            pruned_count = len(pruned_indices)
            reduction_ratio = (original_count - pruned_count) / original_count
            speedup_estimate = 1.0 / factor if factor > 0 else 1.0
            
            result = {
                'factor': factor,
                'original': original_count,
                'pruned': pruned_count,
                'reduction': reduction_ratio,
                'speedup': speedup_estimate,
                'time': prune_time
            }
            
            pruning_results.append(result)
            
            print(f"📉 降采样因子 {factor:.1f}:")
            print(f"    保留点数: {pruned_count:,} / {original_count:,}")
            print(f"    减少比例: {reduction_ratio:.1%}")
            print(f"    理论加速: {speedup_estimate:.1f}x")
            print(f"    剪枝耗时: {prune_time:.3f}秒")
            print()
        
        # 可视化剪枝效果（如果matplotlib可用）
        self._plot_pruning_results(pruning_results)
        
        print("=" * 60)
    
    def _plot_pruning_results(self, results: List[Dict]):
        """可视化剪枝结果"""
        try:
            factors = [r['factor'] for r in results]
            reductions = [r['reduction'] for r in results]
            speedups = [r['speedup'] for r in results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 减少比例
            ax1.plot(factors, reductions, 'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('降采样因子')
            ax1.set_ylabel('点云减少比例')
            ax1.set_title('剪枝效果')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # 理论加速比
            ax2.plot(factors, speedups, 'ro-', linewidth=2, markersize=8)
            ax2.set_xlabel('降采样因子')
            ax2.set_ylabel('理论加速比')
            ax2.set_title('性能提升')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('pruning_analysis.png', dpi=150, bbox_inches='tight')
            print("📊 剪枝分析图已保存为 pruning_analysis.png")
            
        except ImportError:
            print("⚠️  matplotlib不可用，跳过可视化")
    
    async def demonstrate_end_to_end_search(self):
        """演示端到端搜索流程"""
        print("🔍 端到端智能搜索演示")
        print("=" * 60)
        
        for i, demo in enumerate(self.demo_queries, 1):
            print(f"\n🔎 搜索 {i}: \"{demo['query']}\"")
            print(f"期望类型: {demo['description']}")
            print("-" * 40)
            
            # 执行完整搜索
            start_time = time.time()
            results = await self.rag_system.intelligent_search(
                query=demo['query'],
                top_k=5,
                downsample_factor=0.3
            )
            search_time = time.time() - start_time
            
            # 分析结果
            intent = results['intent']
            metrics = results['performance_metrics']
            
            print(f"⏱️  搜索耗时: {results['processing_time']:.3f}秒")
            print(f"📊 性能指标:")
            print(f"    原始高斯点: {metrics['total_gaussians']:,}")
            print(f"    剪枝后: {metrics['pruned_gaussians']:,} ({metrics['pruning_ratio']:.1%})")
            print(f"    初始候选: {metrics['initial_candidates']}")
            print(f"    最终结果: {metrics['final_results']}")
            print(f"    理论加速: {metrics['speedup_estimate']:.1f}x")
            
            print(f"🎯 查询意图:")
            print(f"    类型: {intent['query_type']}")
            print(f"    置信度: {intent['confidence']:.3f}")
            print(f"    权重: {intent['priority_weights']}")
            
            print(f"🏆 前3个结果:")
            for j, result in enumerate(results['final_results'][:3], 1):
                print(f"    {j}. 聚类{result['cluster_id']} (得分: {result['final_score']:.3f})")
                print(f"       - 向量相似度: {result['vector_similarity']:.3f}")
                print(f"       - 文本相似度: {result['text_similarity']:.3f}")
                print(f"       - 视觉相似度: {result['visual_similarity']:.3f}")
                print(f"       - 空间相关性: {result['spatial_relevance']:.3f}")
                print(f"       - 多视角一致性: {result['multi_view_consistency']:.3f}")
            
            print()
    
    async def demonstrate_performance_comparison(self):
        """演示性能对比"""
        print("⚡ 性能对比演示")
        print("=" * 60)
        
        test_queries = [
            "红色椅子",
            "客厅家具", 
            "现代装饰"
        ]
        
        # 传统方法 vs 智能剪枝方法
        print("对比传统全量搜索 vs 智能剪枝搜索:")
        print()
        
        for query in test_queries:
            print(f"查询: \"{query}\"")
            
            # 模拟传统全量搜索
            traditional_time = self._simulate_traditional_search()
            
            # 智能剪枝搜索
            start_time = time.time()
            results = await self.rag_system.intelligent_search(query, top_k=5)
            intelligent_time = time.time() - start_time
            
            speedup = traditional_time / intelligent_time
            memory_reduction = 1 - results['performance_metrics']['pruning_ratio']
            
            print(f"  传统方法: {traditional_time:.3f}秒")
            print(f"  智能方法: {intelligent_time:.3f}秒")
            print(f"  性能提升: {speedup:.1f}x")
            print(f"  内存节省: {memory_reduction:.1%}")
            print()
        
        print("=" * 60)
    
    def _simulate_traditional_search(self) -> float:
        """模拟传统全量搜索时间"""
        # 基于模型大小估算传统搜索时间
        num_gaussians = len(self.rag_system.model_info.means)
        # 假设每1000个高斯点需要0.01秒处理
        return (num_gaussians / 1000) * 0.01 + 0.5  # 基础开销0.5秒
    
    async def demonstrate_caching_effects(self):
        """演示缓存效果"""
        print("💾 缓存效果演示")
        print("=" * 60)
        
        test_query = "客厅的红色沙发"
        
        print(f"测试查询: \"{test_query}\"")
        print()
        
        # 第一次搜索（无缓存）
        print("🔄 第一次搜索（冷启动）:")
        start_time = time.time()
        results1 = await self.rag_system.intelligent_search(test_query)
        first_time = time.time() - start_time
        print(f"  耗时: {first_time:.3f}秒")
        
        # 第二次搜索（命中缓存）
        print("⚡ 第二次搜索（命中缓存）:")
        start_time = time.time()
        results2 = await self.rag_system.intelligent_search(test_query)
        cached_time = time.time() - start_time
        print(f"  耗时: {cached_time:.3f}秒")
        
        cache_speedup = first_time / max(cached_time, 0.001)
        print(f"🚀 缓存加速: {cache_speedup:.1f}x")
        
        # 显示缓存统计
        stats = self.rag_system.get_system_stats()
        print(f"📈 缓存统计:")
        print(f"  意图缓存: {stats['cached_intents']} 条")
        print(f"  结果缓存: {stats['cached_results']} 条")
        
        print("\n" + "=" * 60)
    
    async def demonstrate_query_variations(self):
        """演示查询变化的动态适应"""
        print("🔄 查询变化动态适应演示")
        print("=" * 60)
        
        # 相似查询的变化
        query_variations = [
            "红色椅子",
            "红色的椅子",
            "找一把红色椅子",
            "我想要红色的椅子",
            "有没有红色椅子"
        ]
        
        print("测试相似查询的意图理解一致性:")
        print()
        
        intent_results = []
        
        for query in query_variations:
            intent = await self.rag_system._analyze_query_intent(query)
            intent_results.append({
                'query': query,
                'type': intent.query_type.value,
                'objects': intent.primary_objects,
                'confidence': intent.confidence
            })
            
            print(f"  \"{query}\"")
            print(f"    类型: {intent.query_type.value}")
            print(f"    对象: {intent.primary_objects}")
            print(f"    置信度: {intent.confidence:.3f}")
            print()
        
        # 分析一致性
        types = [r['type'] for r in intent_results]
        type_consistency = len(set(types)) == 1
        
        print(f"🎯 意图识别一致性: {'✅ 一致' if type_consistency else '❌ 不一致'}")
        
        print("\n" + "=" * 60)
    
    async def run_full_demo(self, provider_type: str = "local", api_key: str = None):
        """运行完整演示"""
        print("🌟 智能RAG动态数据库完整演示")
        print("=" * 80)
        print()
        
        try:
            # 初始化系统
            await self.initialize_system(provider_type, api_key)
            
            # 逐个演示各个功能
            await self.demonstrate_intent_analysis()
            await self.demonstrate_dynamic_pruning()
            await self.demonstrate_end_to_end_search()
            await self.demonstrate_performance_comparison()
            await self.demonstrate_caching_effects()
            await self.demonstrate_query_variations()
            
            # 最终统计
            print("📊 最终系统统计")
            print("=" * 60)
            final_stats = self.rag_system.get_system_stats()
            print(f"模型规模: {final_stats['model_gaussians']:,} 个高斯点")
            print(f"意图缓存: {final_stats['cached_intents']} 条")
            print(f"结果缓存: {final_stats['cached_results']} 条")
            print(f"向量索引: {final_stats['vector_index_size']} 个向量")
            
            print("\n🎉 演示完成！智能RAG系统展示了以下核心能力:")
            print("  ✅ LLM驱动的查询意图理解")
            print("  ✅ 基于意图的动态模型剪枝")
            print("  ✅ 多因子相似度计算和重排序")
            print("  ✅ 缓存机制实现的性能优化")
            print("  ✅ 类似动态数据库的高效检索")
            
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """主函数"""
    print("智能RAG动态数据库演示程序")
    print("=" * 50)
    
    # 检查命令行参数
    provider_type = "local"  # 默认使用本地提供商
    api_key = None
    
    if len(sys.argv) > 1:
        provider_type = sys.argv[1].lower()
    
    if len(sys.argv) > 2:
        api_key = sys.argv[2]
    
    # 验证提供商类型
    if provider_type not in ["local", "openai", "claude"]:
        print("❌ 不支持的提供商类型，使用 'local', 'openai', 或 'claude'")
        return
    
    if provider_type in ["openai", "claude"] and not api_key:
        print("⚠️  远程提供商需要API密钥")
        print("使用方法:")
        print(f"  python {sys.argv[0]} openai YOUR_API_KEY")
        print(f"  python {sys.argv[0]} claude YOUR_API_KEY")
        print("或者使用本地提供商:")
        print(f"  python {sys.argv[0]} local")
        return
    
    # 创建并运行演示
    demo = DynamicDatabaseDemo()
    await demo.run_full_demo(provider_type, api_key)

if __name__ == "__main__":
    asyncio.run(main())