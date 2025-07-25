#!/usr/bin/env python3
"""
实际应用示例 - 智能RAG系统的实际部署和使用模式
展示如何在生产环境中集成和使用智能化3D场景检索系统
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SceneRAGApplication:
    """智能场景RAG应用类 - 生产环境示例"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化应用
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.rag_system = None
        self.session_cache = {}
        self.query_history = []
        self.performance_metrics = []
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        default_config = {
            "model_path": "./model/scene.ply",
            "llm_provider": {
                "type": "local",
                "model": "mistral-7b-instruct",
                "api_key": None
            },
            "vector_db_path": "./data/vectors.index",
            "cache_settings": {
                "max_intent_cache": 1000,
                "max_result_cache": 500,
                "cache_ttl": 3600
            },
            "search_settings": {
                "default_top_k": 10,
                "default_downsample": 0.3,
                "max_top_k": 50
            },
            "performance": {
                "enable_metrics": True,
                "log_slow_queries": True,
                "slow_query_threshold": 2.0
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并配置
                config = {**default_config, **user_config}
                logger.info(f"配置文件加载成功: {config_path}")
            else:
                config = default_config
                logger.info("使用默认配置")
            
            return config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return default_config
    
    async def initialize(self):
        """初始化RAG系统"""
        logger.info("初始化智能RAG应用...")
        
        try:
            # 导入RAG模块
            from intelligent_rag import create_intelligent_rag
            
            # 创建RAG系统
            llm_config = self.config["llm_provider"]
            self.rag_system = create_intelligent_rag(
                model_path=self.config["model_path"],
                provider_type=llm_config["type"],
                api_key=llm_config.get("api_key"),
                vector_db_path=self.config.get("vector_db_path"),
                model=llm_config.get("model")
            )
            
            logger.info("✅ RAG系统初始化完成")
            
            # 预热系统
            await self._warmup_system()
            
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            raise
    
    async def _warmup_system(self):
        """预热系统"""
        logger.info("预热系统中...")
        
        warmup_queries = [
            "测试查询",
            "椅子",
            "桌子"
        ]
        
        for query in warmup_queries:
            try:
                await self.rag_system.intelligent_search(
                    query=query,
                    top_k=3,
                    downsample_factor=0.5
                )
            except Exception as e:
                logger.warning(f"预热查询失败: {query} - {e}")
        
        logger.info("✅ 系统预热完成")
    
    async def search(self, query: str, user_id: str = "anonymous", 
                    session_id: str = None, **kwargs) -> Dict:
        """
        执行智能搜索
        
        Args:
            query: 搜索查询
            user_id: 用户ID
            session_id: 会话ID
            **kwargs: 其他搜索参数
            
        Returns:
            搜索结果
        """
        start_time = time.time()
        
        # 参数验证和默认值
        top_k = min(
            kwargs.get('top_k', self.config["search_settings"]["default_top_k"]),
            self.config["search_settings"]["max_top_k"]
        )
        
        downsample_factor = kwargs.get(
            'downsample_factor', 
            self.config["search_settings"]["default_downsample"]
        )
        
        # 记录查询
        query_record = {
            "timestamp": start_time,
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "parameters": {"top_k": top_k, "downsample_factor": downsample_factor}
        }
        
        try:
            logger.info(f"用户 {user_id} 搜索: \"{query}\"")
            
            # 执行搜索
            results = await self.rag_system.intelligent_search(
                query=query,
                top_k=top_k,
                downsample_factor=downsample_factor
            )
            
            # 处理结果
            processed_results = self._process_search_results(results, query_record)
            
            # 记录性能指标
            self._record_performance_metrics(query_record, processed_results)
            
            # 添加到查询历史
            self.query_history.append(query_record)
            
            # 限制历史记录大小
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]
            
            logger.info(f"搜索完成，耗时 {processed_results['total_time']:.3f}秒")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            
            # 返回错误结果
            error_result = {
                "success": False,
                "error": str(e),
                "query": query,
                "timestamp": start_time,
                "total_time": time.time() - start_time
            }
            
            return error_result
    
    def _process_search_results(self, raw_results: Dict, query_record: Dict) -> Dict:
        """处理搜索结果"""
        total_time = time.time() - query_record["timestamp"]
        
        # 增强结果信息
        processed_results = {
            "success": True,
            "query": raw_results["query"],
            "timestamp": query_record["timestamp"],
            "total_time": total_time,
            "user_id": query_record["user_id"],
            "session_id": query_record["session_id"],
            
            # 搜索结果
            "results": raw_results["final_results"],
            "result_count": len(raw_results["final_results"]),
            
            # 查询意图
            "intent": raw_results["intent"],
            
            # 性能信息
            "performance": {
                "processing_time": raw_results["processing_time"],
                "pruned_gaussians": raw_results["pruned_gaussians"],
                "initial_candidates": raw_results["initial_candidates"],
                "metrics": raw_results["performance_metrics"]
            },
            
            # 系统信息
            "system_info": {
                "total_gaussians": raw_results["performance_metrics"]["total_gaussians"],
                "pruning_ratio": raw_results["performance_metrics"]["pruning_ratio"],
                "speedup_estimate": raw_results["performance_metrics"]["speedup_estimate"]
            }
        }
        
        return processed_results
    
    def _record_performance_metrics(self, query_record: Dict, results: Dict):
        """记录性能指标"""
        if not self.config["performance"]["enable_metrics"]:
            return
        
        metric = {
            "timestamp": query_record["timestamp"],
            "query_length": len(query_record["query"]),
            "total_time": results["total_time"],
            "processing_time": results["performance"]["processing_time"],
            "result_count": results["result_count"],
            "pruning_ratio": results["performance"]["metrics"]["pruning_ratio"],
            "intent_confidence": results["intent"]["confidence"],
            "query_type": results["intent"]["query_type"]
        }
        
        self.performance_metrics.append(metric)
        
        # 慢查询日志
        if (self.config["performance"]["log_slow_queries"] and 
            results["total_time"] > self.config["performance"]["slow_query_threshold"]):
            logger.warning(
                f"慢查询检测: \"{query_record['query']}\" "
                f"耗时 {results['total_time']:.3f}秒"
            )
        
        # 限制指标历史大小
        if len(self.performance_metrics) > 10000:
            self.performance_metrics = self.performance_metrics[-10000:]
    
    async def batch_search(self, queries: List[str], user_id: str = "batch_user") -> List[Dict]:
        """批量搜索"""
        logger.info(f"批量搜索开始，共 {len(queries)} 个查询")
        
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"处理查询 {i+1}/{len(queries)}: \"{query}\"")
            
            try:
                result = await self.search(
                    query=query,
                    user_id=user_id,
                    session_id=f"batch_{int(time.time())}"
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"批量搜索中的查询失败: {query} - {e}")
                results.append({
                    "success": False,
                    "query": query,
                    "error": str(e)
                })
        
        logger.info(f"批量搜索完成，成功 {sum(1 for r in results if r.get('success', False))} 个")
        
        return results
    
    def get_analytics(self) -> Dict:
        """获取分析数据"""
        if not self.performance_metrics:
            return {"message": "暂无性能数据"}
        
        metrics = self.performance_metrics
        
        analytics = {
            "total_queries": len(metrics),
            "time_range": {
                "start": min(m["timestamp"] for m in metrics),
                "end": max(m["timestamp"] for m in metrics)
            },
            "performance": {
                "avg_total_time": np.mean([m["total_time"] for m in metrics]),
                "avg_processing_time": np.mean([m["processing_time"] for m in metrics]),
                "median_total_time": np.median([m["total_time"] for m in metrics]),
                "p95_total_time": np.percentile([m["total_time"] for m in metrics], 95),
                "max_total_time": max(m["total_time"] for m in metrics)
            },
            "query_stats": {
                "avg_result_count": np.mean([m["result_count"] for m in metrics]),
                "avg_query_length": np.mean([m["query_length"] for m in metrics]),
                "avg_pruning_ratio": np.mean([m["pruning_ratio"] for m in metrics])
            },
            "intent_distribution": self._analyze_intent_distribution(metrics),
            "recent_queries": self.query_history[-10:] if self.query_history else []
        }
        
        return analytics
    
    def _analyze_intent_distribution(self, metrics: List[Dict]) -> Dict:
        """分析意图分布"""
        intent_counts = {}
        confidence_by_type = {}
        
        for metric in metrics:
            query_type = metric["query_type"]
            confidence = metric["intent_confidence"]
            
            intent_counts[query_type] = intent_counts.get(query_type, 0) + 1
            
            if query_type not in confidence_by_type:
                confidence_by_type[query_type] = []
            confidence_by_type[query_type].append(confidence)
        
        # 计算统计信息
        distribution = {}
        for query_type, count in intent_counts.items():
            distribution[query_type] = {
                "count": count,
                "percentage": count / len(metrics) * 100,
                "avg_confidence": np.mean(confidence_by_type[query_type]),
                "min_confidence": min(confidence_by_type[query_type]),
                "max_confidence": max(confidence_by_type[query_type])
            }
        
        return distribution
    
    def export_analytics(self, filepath: str = "analytics.json"):
        """导出分析数据"""
        analytics = self.get_analytics()
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"分析数据已导出到: {filepath}")
            
        except Exception as e:
            logger.error(f"导出分析数据失败: {e}")
    
    def clear_cache(self):
        """清空缓存"""
        if self.rag_system:
            self.rag_system.clear_cache()
        
        self.session_cache.clear()
        logger.info("应用缓存已清空")
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        if not self.rag_system:
            return {"status": "not_initialized"}
        
        rag_stats = self.rag_system.get_system_stats()
        
        status = {
            "status": "running",
            "uptime": time.time() - (self.query_history[0]["timestamp"] if self.query_history else time.time()),
            "total_queries": len(self.query_history),
            "cache_info": {
                "intent_cache": rag_stats["cached_intents"],
                "result_cache": rag_stats["cached_results"],
                "session_cache": len(self.session_cache)
            },
            "model_info": {
                "total_gaussians": rag_stats["model_gaussians"],
                "vector_index_size": rag_stats["vector_index_size"]
            },
            "config": self.config
        }
        
        return status

class WebAPIWrapper:
    """Web API包装器示例（用于生产环境集成）"""
    
    def __init__(self, app: SceneRAGApplication):
        self.app = app
    
    async def handle_search_request(self, request_data: Dict) -> Dict:
        """处理搜索请求"""
        try:
            # 提取请求参数
            query = request_data.get("query", "").strip()
            user_id = request_data.get("user_id", "anonymous")
            session_id = request_data.get("session_id")
            top_k = request_data.get("top_k", 10)
            downsample_factor = request_data.get("downsample_factor", 0.3)
            
            # 参数验证
            if not query:
                return {
                    "success": False,
                    "error": "查询不能为空",
                    "error_code": "EMPTY_QUERY"
                }
            
            if len(query) > 500:
                return {
                    "success": False,
                    "error": "查询长度超过限制（500字符）",
                    "error_code": "QUERY_TOO_LONG"
                }
            
            # 执行搜索
            results = await self.app.search(
                query=query,
                user_id=user_id,
                session_id=session_id,
                top_k=top_k,
                downsample_factor=downsample_factor
            )
            
            return results
            
        except Exception as e:
            logger.error(f"API请求处理失败: {e}")
            return {
                "success": False,
                "error": "内部服务器错误",
                "error_code": "INTERNAL_ERROR"
            }
    
    async def handle_analytics_request(self) -> Dict:
        """处理分析请求"""
        try:
            return self.app.get_analytics()
        except Exception as e:
            logger.error(f"分析请求处理失败: {e}")
            return {
                "success": False,
                "error": "获取分析数据失败"
            }
    
    async def handle_status_request(self) -> Dict:
        """处理状态请求"""
        try:
            return self.app.get_system_status()
        except Exception as e:
            logger.error(f"状态请求处理失败: {e}")
            return {
                "success": False,
                "error": "获取系统状态失败"
            }

async def example_usage():
    """使用示例"""
    logger.info("智能RAG应用使用示例")
    
    # 创建应用实例
    app = SceneRAGApplication()
    
    try:
        # 初始化
        await app.initialize()
        
        # 单个搜索示例
        print("\n=== 单个搜索示例 ===")
        result = await app.search(
            query="红色的现代椅子",
            user_id="user123",
            session_id="session456",
            top_k=5
        )
        
        print(f"搜索结果: {result['result_count']} 个")
        print(f"查询类型: {result['intent']['query_type']}")
        print(f"处理时间: {result['total_time']:.3f}秒")
        
        # 批量搜索示例
        print("\n=== 批量搜索示例 ===")
        batch_queries = [
            "客厅的桌子",
            "现代风格装饰",
            "舒适的沙发"
        ]
        
        batch_results = await app.batch_search(batch_queries, "batch_user")
        print(f"批量搜索完成: {len(batch_results)} 个查询")
        
        # 分析数据示例
        print("\n=== 分析数据示例 ===")
        analytics = app.get_analytics()
        print(f"总查询数: {analytics['total_queries']}")
        print(f"平均处理时间: {analytics['performance']['avg_total_time']:.3f}秒")
        
        # 系统状态示例
        print("\n=== 系统状态示例 ===")
        status = app.get_system_status()
        print(f"系统状态: {status['status']}")
        print(f"模型规模: {status['model_info']['total_gaussians']:,} 个高斯点")
        
        # Web API示例
        print("\n=== Web API示例 ===")
        api = WebAPIWrapper(app)
        
        api_request = {
            "query": "寻找蓝色装饰品",
            "user_id": "api_user",
            "top_k": 3
        }
        
        api_result = await api.handle_search_request(api_request)
        print(f"API搜索成功: {api_result.get('success', False)}")
        
        # 导出分析数据
        app.export_analytics("example_analytics.json")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")

if __name__ == "__main__":
    # 运行使用示例
    asyncio.run(example_usage())