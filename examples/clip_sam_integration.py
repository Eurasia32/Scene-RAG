#!/usr/bin/env python3
"""
GSRender + CLIP + SAM 集成示例
展示如何将3D高斯溅射渲染与CLIP特征提取和SAM语义分割结合
"""

import numpy as np
import torch
import cv2
import sys
import os
from typing import List, Dict, Tuple, Optional
import json

# 导入GSRender模块
try:
    import gsrender
    print("✓ GSRender模块导入成功")
except ImportError as e:
    print(f"✗ GSRender模块导入失败: {e}")
    print("请先编译Python扩展: python setup.py build_ext --inplace")
    sys.exit(1)

# 可选的CLIP和SAM导入
CLIP_AVAILABLE = False
SAM_AVAILABLE = False

try:
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
    print("✓ CLIP模块可用")
except ImportError:
    print("○ CLIP模块不可用 (pip install git+https://github.com/openai/CLIP.git)")

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
    print("✓ SAM模块可用")
except ImportError:
    print("○ SAM模块不可用 (pip install git+https://github.com/facebookresearch/segment-anything.git)")

class GSRenderCLIPSAMIntegration:
    """GSRender + CLIP + SAM 集成类"""
    
    def __init__(self, ply_path: str, device: str = "auto"):
        """
        初始化集成系统
        
        Args:
            ply_path: PLY模型文件路径
            device: 计算设备 ("auto", "cpu", "cuda")
        """
        self.device = self._get_device(device)
        print(f"使用设备: {self.device}")
        
        # 初始化GSRender
        self.renderer = gsrender.GSRenderInterface()
        self.renderer.set_background_color(1.0, 1.0, 1.0)  # 白色背景
        
        success = self.renderer.load_model(ply_path, self.device)
        if not success:
            raise RuntimeError(f"无法加载模型: {ply_path}")
        
        self.model_info = self.renderer.get_model_info()
        print(f"✓ 模型加载成功: {self.model_info.num_gaussians} 高斯点")
        
        # 初始化CLIP
        self.clip_model = None
        self.clip_preprocess = None
        if CLIP_AVAILABLE:
            self._init_clip()
        
        # 初始化SAM
        self.sam_predictor = None
        if SAM_AVAILABLE:
            self._init_sam()
    
    def _get_device(self, device: str) -> str:
        """确定计算设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _init_clip(self):
        """初始化CLIP模型"""
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("✓ CLIP模型初始化成功")
        except Exception as e:
            print(f"✗ CLIP模型初始化失败: {e}")
            self.clip_model = None
    
    def _init_sam(self, sam_checkpoint: str = ""):
        """初始化SAM模型"""
        if not sam_checkpoint:
            # 尝试找到SAM检查点文件
            sam_paths = [
                "sam_vit_h_4b8939.pth",
                "models/sam_vit_h_4b8939.pth",
                os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth")
            ]
            
            for path in sam_paths:
                if os.path.exists(path):
                    sam_checkpoint = path
                    break
        
        if not sam_checkpoint or not os.path.exists(sam_checkpoint):
            print("○ SAM检查点文件未找到，跳过SAM初始化")
            print("  请下载SAM模型: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            return
        
        try:
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print("✓ SAM模型初始化成功")
        except Exception as e:
            print(f"✗ SAM模型初始化失败: {e}")
            self.sam_predictor = None
    
    def render_with_camera_pose(self, camera_pose: np.ndarray, 
                               width: int = 800, height: int = 600,
                               fx: float = 400.0, fy: float = 400.0,
                               gaussian_indices: Optional[List[int]] = None) -> gsrender.RenderResult:
        """
        根据相机位姿渲染场景
        
        Args:
            camera_pose: 4x4相机位姿矩阵 (相机到世界坐标)
            width, height: 图像尺寸
            fx, fy: 相机焦距
            gaussian_indices: 可选的高斯点索引列表
            
        Returns:
            渲染结果
        """
        camera_params = gsrender.create_camera_params(
            view_matrix=camera_pose.astype(np.float32),
            width=width,
            height=height,
            fx=fx,
            fy=fy
        )
        
        return self.renderer.render(camera_params, gaussian_indices)
    
    def extract_clip_features(self, rgb_image: np.ndarray, 
                             masks: Optional[List[np.ndarray]] = None) -> Dict:
        """
        提取CLIP特征
        
        Args:
            rgb_image: RGB图像 [H, W, 3]
            masks: 可选的分割掩码列表
            
        Returns:
            包含特征的字典
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            return {"error": "CLIP不可用"}
        
        features = {}
        
        try:
            # 全图特征
            pil_image = Image.fromarray((rgb_image * 255).astype(np.uint8))
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                features["full_image"] = image_features.cpu().numpy()
            
            # 区域特征 (如果提供了掩码)
            if masks is not None:
                features["regions"] = []
                
                for i, mask in enumerate(masks):
                    # 创建掩码区域图像
                    masked_image = rgb_image.copy()
                    masked_image[mask == 0] = 1.0  # 背景设为白色
                    
                    pil_masked = Image.fromarray((masked_image * 255).astype(np.uint8))
                    masked_input = self.clip_preprocess(pil_masked).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        region_features = self.clip_model.encode_image(masked_input)
                        features["regions"].append({
                            "region_id": i,
                            "features": region_features.cpu().numpy(),
                            "mask_area": np.sum(mask > 0)
                        })
            
        except Exception as e:
            features["error"] = str(e)
        
        return features
    
    def segment_with_sam(self, rgb_image: np.ndarray, 
                        points: Optional[List[Tuple[int, int]]] = None,
                        point_labels: Optional[List[int]] = None) -> Dict:
        """
        使用SAM进行语义分割
        
        Args:
            rgb_image: RGB图像 [H, W, 3]
            points: 可选的提示点列表 [(x, y), ...]
            point_labels: 点标签列表 (1=前景，0=背景)
            
        Returns:
            分割结果字典
        """
        if not SAM_AVAILABLE or self.sam_predictor is None:
            return {"error": "SAM不可用"}
        
        try:
            # 转换为uint8格式
            image_uint8 = (rgb_image * 255).astype(np.uint8)
            
            # 设置图像
            self.sam_predictor.set_image(image_uint8)
            
            result = {}
            
            if points is not None and len(points) > 0:
                # 点提示分割
                input_points = np.array(points)
                input_labels = np.array(point_labels) if point_labels else np.ones(len(points))
                
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
                
                result["masks"] = masks
                result["scores"] = scores
                result["logits"] = logits
                result["method"] = "point_prompt"
                
            else:
                # 自动分割 (需要额外的工具)
                # 这里可以集成SAM的自动分割功能
                result["error"] = "自动分割需要额外实现"
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def query_by_text(self, text_query: str, camera_poses: List[np.ndarray],
                     top_k: int = 5) -> List[Dict]:
        """
        基于文本查询场景
        
        Args:
            text_query: 文本查询
            camera_poses: 相机位姿列表
            top_k: 返回最相似的top-k个视角
            
        Returns:
            查询结果列表
        """
        if not CLIP_AVAILABLE or self.clip_model is None:
            return [{"error": "CLIP不可用"}]
        
        results = []
        
        try:
            # 编码文本查询
            text_tokens = clip.tokenize([text_query]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarities = []
            
            # 渲染每个视角并计算相似度
            for i, camera_pose in enumerate(camera_poses):
                render_result = self.render_with_camera_pose(camera_pose)
                
                # 提取CLIP特征
                clip_result = self.extract_clip_features(render_result.rgb_image)
                
                if "error" not in clip_result:
                    image_features = torch.from_numpy(clip_result["full_image"]).to(self.device)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # 计算相似度
                    similarity = (text_features @ image_features.T).item()
                    
                    similarities.append({
                        "view_id": i,
                        "similarity": similarity,
                        "render_result": render_result,
                        "camera_pose": camera_pose
                    })
            
            # 按相似度排序并返回top-k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            results = similarities[:top_k]
            
        except Exception as e:
            results = [{"error": str(e)}]
        
        return results
    
    def analyze_gaussian_semantics(self, camera_pose: np.ndarray,
                                  text_queries: List[str]) -> Dict:
        """
        分析高斯点的语义信息
        
        Args:
            camera_pose: 相机位姿
            text_queries: 文本查询列表
            
        Returns:
            语义分析结果
        """
        result = {
            "camera_pose": camera_pose,
            "text_queries": text_queries,
            "semantic_mapping": {}
        }
        
        try:
            # 渲染场景
            render_result = self.render_with_camera_pose(camera_pose)
            
            # SAM分割
            if SAM_AVAILABLE and self.sam_predictor is not None:
                # 使用网格点进行自动分割
                h, w = render_result.height, render_result.width
                grid_points = []
                for y in range(h//4, h, h//4):
                    for x in range(w//4, w, w//4):
                        grid_points.append((x, y))
                
                sam_result = self.segment_with_sam(
                    render_result.rgb_image, 
                    points=grid_points,
                    point_labels=[1] * len(grid_points)
                )
                
                if "masks" in sam_result:
                    masks = sam_result["masks"]
                    
                    # 为每个分割区域提取CLIP特征
                    clip_result = self.extract_clip_features(render_result.rgb_image, masks)
                    
                    if "regions" in clip_result:
                        # 计算与文本查询的相似度
                        for region_info in clip_result["regions"]:
                            region_id = region_info["region_id"]
                            region_features = torch.from_numpy(region_info["features"]).to(self.device)
                            region_features = region_features / region_features.norm(dim=-1, keepdim=True)
                            
                            region_similarities = {}
                            
                            for text_query in text_queries:
                                text_tokens = clip.tokenize([text_query]).to(self.device)
                                with torch.no_grad():
                                    text_features = self.clip_model.encode_text(text_tokens)
                                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                                    
                                    similarity = (text_features @ region_features.T).item()
                                    region_similarities[text_query] = similarity
                            
                            # 获取该区域对应的高斯点
                            mask = masks[region_id]
                            gaussian_ids = set()
                            
                            for y in range(mask.shape[0]):
                                for x in range(mask.shape[1]):
                                    if mask[y, x]:
                                        pixel_gaussians = render_result.get_pixel_gaussians(x, y)
                                        gaussian_ids.update(pixel_gaussians)
                            
                            result["semantic_mapping"][region_id] = {
                                "gaussian_ids": list(gaussian_ids),
                                "text_similarities": region_similarities,
                                "mask_area": region_info["mask_area"],
                                "best_match": max(region_similarities.items(), key=lambda x: x[1])
                            }
            
        except Exception as e:
            result["error"] = str(e)
        
        return result

def demo_basic_integration():
    """基础集成演示"""
    print("\\n=== GSRender + CLIP + SAM 基础集成演示 ===")
    
    # 检查模型文件
    ply_path = "model/model.ply"
    if not os.path.exists(ply_path):
        print(f"✗ 模型文件不存在: {ply_path}")
        return
    
    # 创建集成系统
    system = GSRenderCLIPSAMIntegration(ply_path)
    
    # 创建相机位姿
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # 渲染场景
    print("渲染场景...")
    render_result = system.render_with_camera_pose(camera_pose)
    
    # 保存渲染结果
    rgb_bgr = cv2.cvtColor((render_result.rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("integration_render.png", rgb_bgr)
    print("✓ 渲染结果已保存: integration_render.png")
    
    # CLIP特征提取
    if CLIP_AVAILABLE:
        print("提取CLIP特征...")
        clip_features = system.extract_clip_features(render_result.rgb_image)
        
        if "error" not in clip_features:
            print(f"✓ CLIP特征提取成功，特征维度: {clip_features['full_image'].shape}")
        else:
            print(f"✗ CLIP特征提取失败: {clip_features['error']}")
    
    # SAM分割
    if SAM_AVAILABLE:
        print("执行SAM分割...")
        # 使用图像中心点作为提示
        center_x, center_y = render_result.width // 2, render_result.height // 2
        sam_result = system.segment_with_sam(
            render_result.rgb_image,
            points=[(center_x, center_y)],
            point_labels=[1]
        )
        
        if "error" not in sam_result:
            masks = sam_result["masks"]
            print(f"✓ SAM分割成功，生成 {len(masks)} 个掩码")
            
            # 保存分割结果
            for i, mask in enumerate(masks):
                mask_image = (mask.astype(np.uint8) * 255)
                cv2.imwrite(f"sam_mask_{i}.png", mask_image)
            
            print(f"✓ 分割掩码已保存: sam_mask_*.png")
        else:
            print(f"✗ SAM分割失败: {sam_result['error']}")

def demo_text_query():
    """文本查询演示"""
    print("\\n=== 文本查询演示 ===")
    
    ply_path = "model/model.ply"
    if not os.path.exists(ply_path):
        print(f"✗ 模型文件不存在: {ply_path}")
        return
    
    if not CLIP_AVAILABLE:
        print("✗ 文本查询需要CLIP支持")
        return
    
    system = GSRenderCLIPSAMIntegration(ply_path)
    
    # 创建多个相机位姿
    camera_poses = []
    for i in range(8):
        angle = i * 2 * np.pi / 8
        cam_x = 5.0 * np.cos(angle)
        cam_z = 5.0 * np.sin(angle)
        
        camera_pose = np.array([
            [np.cos(angle), 0.0, np.sin(angle), cam_x],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle), cam_z],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        camera_poses.append(camera_pose)
    
    # 文本查询
    text_queries = [
        "a beautiful scene",
        "modern architecture", 
        "natural landscape",
        "indoor furniture"
    ]
    
    for text_query in text_queries:
        print(f"\\n查询: '{text_query}'")
        results = system.query_by_text(text_query, camera_poses, top_k=3)
        
        if results and "error" not in results[0]:
            print(f"找到 {len(results)} 个匹配视角:")
            
            for i, result in enumerate(results):
                print(f"  视角 {result['view_id']}: 相似度 {result['similarity']:.3f}")
                
                # 保存匹配的视角
                rgb_bgr = cv2.cvtColor(
                    (result['render_result'].rgb_image * 255).astype(np.uint8), 
                    cv2.COLOR_RGB2BGR
                )
                filename = f"query_{text_query.replace(' ', '_')}_view_{i}.png"
                cv2.imwrite(filename, rgb_bgr)
            
            print(f"✓ 查询结果已保存")
        else:
            print(f"✗ 查询失败")

def demo_semantic_analysis():
    """语义分析演示"""
    print("\\n=== 语义分析演示 ===")
    
    ply_path = "model/model.ply"
    if not os.path.exists(ply_path):
        print(f"✗ 模型文件不存在: {ply_path}")
        return
    
    if not (CLIP_AVAILABLE and SAM_AVAILABLE):
        print("✗ 语义分析需要CLIP和SAM支持")
        return
    
    system = GSRenderCLIPSAMIntegration(ply_path)
    
    # 相机位姿
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.8, -0.6, 1.0],
        [0.0, 0.6, 0.8, 4.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # 语义查询列表
    text_queries = [
        "furniture",
        "wall",
        "floor", 
        "ceiling",
        "decoration"
    ]
    
    print("执行语义分析...")
    semantic_result = system.analyze_gaussian_semantics(camera_pose, text_queries)
    
    if "error" not in semantic_result:
        print("✓ 语义分析完成")
        
        # 保存结果到JSON
        # 将numpy数组转换为列表以便JSON序列化
        json_result = {}
        for region_id, region_info in semantic_result["semantic_mapping"].items():
            json_result[str(region_id)] = {
                "gaussian_count": len(region_info["gaussian_ids"]),
                "text_similarities": region_info["text_similarities"],
                "mask_area": int(region_info["mask_area"]),
                "best_match": region_info["best_match"]
            }
        
        with open("semantic_analysis.json", "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        
        print("✓ 语义分析结果已保存: semantic_analysis.json")
        
        # 打印摘要
        print("\\n语义分析摘要:")
        for region_id, region_info in semantic_result["semantic_mapping"].items():
            best_match = region_info["best_match"]
            print(f"  区域 {region_id}: {best_match[0]} (相似度: {best_match[1]:.3f})")
    else:
        print(f"✗ 语义分析失败: {semantic_result['error']}")

if __name__ == "__main__":
    print("GSRender + CLIP + SAM 集成示例")
    print("=" * 60)
    
    try:
        demo_basic_integration()
        demo_text_query()
        demo_semantic_analysis()
        
        print("\\n" + "=" * 60)
        print("✓ 所有集成示例运行完成!")
        print("\\n生成的文件:")
        print("  - integration_render.png: 集成渲染结果")
        print("  - sam_mask_*.png: SAM分割掩码")
        print("  - query_*_view_*.png: 文本查询结果")
        print("  - semantic_analysis.json: 语义分析结果")
        
    except Exception as e:
        print(f"\\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()