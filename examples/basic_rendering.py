#!/usr/bin/env python3
"""
GSRender Python接口示例 - 基础渲染功能演示
展示如何使用GSRender进行3D Gaussian Splatting渲染
"""

import numpy as np
import torch
import cv2
import sys
import os

# 导入GSRender模块
try:
    import gsrender
    print("✓ GSRender模块导入成功")
except ImportError as e:
    print(f"✗ GSRender模块导入失败: {e}")
    print("请先编译Python扩展: python setup.py build_ext --inplace")
    sys.exit(1)

def create_sample_camera_params():
    """创建示例相机参数"""
    
    # 创建4x4视图矩阵 (相机到世界坐标变换)
    view_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 5.0],  # 相机在Z=5位置
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # 使用工具函数创建相机参数
    camera_params = gsrender.create_camera_params(
        view_matrix=view_matrix,
        width=800,
        height=600, 
        fx=400.0,
        fy=400.0,
        downscale_factor=1.0,
        sh_degree=3
    )
    
    return camera_params

def basic_rendering_example():
    """基础渲染示例"""
    print("\\n=== 基础渲染示例 ===")
    
    # 创建渲染器
    renderer = gsrender.GSRenderInterface()
    
    # 设置背景颜色为白色
    renderer.set_background_color(1.0, 1.0, 1.0)
    
    # 加载模型 (需要替换为实际的PLY文件路径)
    ply_path = "model/model.ply"  # 请替换为实际路径
    
    if not os.path.exists(ply_path):
        print(f"✗ 模型文件不存在: {ply_path}")
        print("请将PLY文件放置在指定路径或修改ply_path变量")
        return
    
    # 检测设备并加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    success = renderer.load_model(ply_path, device)
    if not success:
        print("✗ 模型加载失败")
        return
    print("✓ 模型加载成功")
    
    # 获取模型信息
    model_info = renderer.get_model_info()
    print(f"模型信息: {model_info}")
    print(f"高斯点数量: {model_info.num_gaussians}")
    print(f"球谐函数阶数: {model_info.sh_degree}")
    
    # 创建相机参数
    camera_params = create_sample_camera_params()
    print(f"相机参数: {camera_params}")
    
    # GPU预热
    if device == "cuda":
        print("GPU预热中...")
        renderer.warmup(camera_params)
    
    # 执行渲染
    print("开始渲染...")
    result = renderer.render(camera_params)
    
    print(f"渲染结果: {result}")
    print(f"渲染时间: {result.render_time_ms:.2f}ms")
    print(f"可见高斯点数: {result.visible_gaussians}")
    
    # 保存RGB图像
    rgb_image = result.rgb_image
    # 转换为BGR格式并保存
    rgb_bgr = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_basic.png", rgb_bgr)
    print("✓ RGB图像已保存: output_basic.png")
    
    # 保存深度图
    depth_image = result.depth_image.squeeze()  # 移除channel维度
    depth_normalized = ((depth_image - depth_image.min()) / 
                       (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)
    cv2.imwrite("output_depth.png", depth_normalized)
    print("✓ 深度图已保存: output_depth.png")
    
    # 显示px2gid信息
    print(f"px2gid映射统计:")
    total_mappings = sum(len(pixel_gaussians) for pixel_gaussians in result.pixel_to_gaussian_mapping)
    print(f"  总映射数: {total_mappings}")
    print(f"  平均每像素高斯点数: {total_mappings / (result.width * result.height):.2f}")
    
    # 查看中心像素的高斯点
    center_x, center_y = result.width // 2, result.height // 2
    center_gaussians = result.get_pixel_gaussians(center_x, center_y)
    print(f"  中心像素({center_x}, {center_y})的高斯点: {center_gaussians[:5]}...")  # 只显示前5个
    
    return renderer, result

def gaussian_filtering_example():
    """高斯点过滤示例"""
    print("\\n=== 高斯点过滤示例 ===")
    
    renderer = gsrender.GSRenderInterface()
    
    # 加载模型
    ply_path = "model/model.ply"
    if not os.path.exists(ply_path):
        print(f"✗ 模型文件不存在: {ply_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer.load_model(ply_path, device)
    
    # 获取模型信息
    model_info = renderer.get_model_info()
    print(f"原始高斯点数量: {model_info.num_gaussians}")
    
    # 1. 按不透明度过滤
    high_opacity_indices = renderer.filter_gaussians(
        filter_by_opacity=True,
        min_opacity=0.5
    )
    print(f"高不透明度高斯点数量 (>0.5): {len(high_opacity_indices)}")
    
    # 2. 按位置过滤 (假设场景在原点附近)
    position_bounds = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]  # [min_x, min_y, min_z, max_x, max_y, max_z]
    center_region_indices = renderer.filter_gaussians(
        filter_by_position=True,
        position_bounds=position_bounds
    )
    print(f"中心区域高斯点数量: {len(center_region_indices)}")
    
    # 3. 组合过滤
    filtered_indices = renderer.filter_gaussians(
        filter_by_position=True,
        position_bounds=position_bounds,
        filter_by_opacity=True,
        min_opacity=0.3
    )
    print(f"组合过滤后高斯点数量: {len(filtered_indices)}")
    
    # 使用过滤后的高斯点进行渲染
    camera_params = create_sample_camera_params()
    
    print("渲染过滤后的场景...")
    filtered_result = renderer.render(camera_params, filtered_indices)
    
    # 保存过滤后的渲染结果
    rgb_filtered = (filtered_result.rgb_image * 255).astype(np.uint8)
    rgb_bgr_filtered = cv2.cvtColor(rgb_filtered, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_filtered.png", rgb_bgr_filtered)
    print("✓ 过滤渲染结果已保存: output_filtered.png")
    
    print(f"过滤后可见高斯点数: {filtered_result.visible_gaussians}")
    print(f"过滤后渲染时间: {filtered_result.render_time_ms:.2f}ms")

def batch_rendering_example():
    """批量渲染示例 - 多视角渲染"""
    print("\\n=== 批量渲染示例 ===")
    
    renderer = gsrender.GSRenderInterface()
    
    # 加载模型
    ply_path = "model/model.ply"
    if not os.path.exists(ply_path):
        print(f"✗ 模型文件不存在: {ply_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer.load_model(ply_path, device)
    
    # 创建多个视角的相机参数
    camera_params_list = []
    
    # 围绕场景创建5个不同角度的视角
    for i in range(5):
        angle = i * 2 * np.pi / 5  # 72度间隔
        
        # 计算相机位置 (半径为5的圆)
        cam_x = 5.0 * np.cos(angle)
        cam_z = 5.0 * np.sin(angle)
        cam_y = 0.0
        
        # 创建朝向原点的视图矩阵
        view_matrix = np.array([
            [np.cos(angle), 0.0, np.sin(angle), cam_x],
            [0.0, 1.0, 0.0, cam_y],
            [-np.sin(angle), 0.0, np.cos(angle), cam_z],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        camera_params = gsrender.create_camera_params(
            view_matrix=view_matrix,
            width=640,
            height=480,
            fx=320.0, 
            fy=320.0,
            downscale_factor=1.0
        )
        
        camera_params_list.append(camera_params)
    
    print(f"准备渲染 {len(camera_params_list)} 个视角...")
    
    # 批量渲染
    results = renderer.render_batch(camera_params_list)
    
    # 保存所有渲染结果
    for i, result in enumerate(results):
        rgb_image = (result.rgb_image * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"output_view_{i:02d}.png", rgb_bgr)
        
        print(f"视角 {i}: {result.visible_gaussians} 高斯点, {result.render_time_ms:.2f}ms")
    
    print(f"✓ 批量渲染完成，保存了 {len(results)} 张图像")
    
    # 计算总渲染时间
    total_time = sum(result.render_time_ms for result in results)
    print(f"总渲染时间: {total_time:.2f}ms")
    print(f"平均单帧时间: {total_time/len(results):.2f}ms")

def px2gid_analysis_example():
    """px2gid映射分析示例"""
    print("\\n=== px2gid映射分析示例 ===")
    
    renderer = gsrender.GSRenderInterface()
    
    # 加载模型并渲染
    ply_path = "model/model.ply"
    if not os.path.exists(ply_path):
        print(f"✗ 模型文件不存在: {ply_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer.load_model(ply_path, device)
    
    camera_params = create_sample_camera_params()
    result = renderer.render(camera_params)
    
    # 分析px2gid映射
    print(f"图像尺寸: {result.width}x{result.height}")
    
    # 统计每个像素的高斯点数量
    gaussians_per_pixel = [len(pixel_gaussians) for pixel_gaussians in result.pixel_to_gaussian_mapping]
    
    print(f"px2gid统计:")
    print(f"  最小高斯点数/像素: {min(gaussians_per_pixel)}")
    print(f"  最大高斯点数/像素: {max(gaussians_per_pixel)}")
    print(f"  平均高斯点数/像素: {np.mean(gaussians_per_pixel):.2f}")
    print(f"  标准差: {np.std(gaussians_per_pixel):.2f}")
    
    # 转换为密集tensor格式
    px2gid_tensor = result.get_px2gid_tensor(max_gaussians_per_pixel=20)
    print(f"px2gid tensor形状: {px2gid_tensor.shape}")
    
    # 创建每像素高斯点数量的热图
    gaussians_count_map = np.array(gaussians_per_pixel).reshape(result.height, result.width)
    
    # 归一化并转换为彩色热图
    normalized_count = ((gaussians_count_map - gaussians_count_map.min()) / 
                       (gaussians_count_map.max() - gaussians_count_map.min()) * 255).astype(np.uint8)
    
    # 应用颜色映射
    heatmap = cv2.applyColorMap(normalized_count, cv2.COLORMAP_JET)
    cv2.imwrite("px2gid_heatmap.png", heatmap)
    print("✓ px2gid热图已保存: px2gid_heatmap.png")
    
    # 分析特定区域的高斯点分布
    center_region = gaussians_count_map[result.height//4:3*result.height//4, 
                                       result.width//4:3*result.width//4]
    print(f"中心区域统计:")
    print(f"  平均高斯点数: {np.mean(center_region):.2f}")
    print(f"  最大高斯点数: {np.max(center_region)}")

if __name__ == "__main__":
    print("GSRender Python接口示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        basic_rendering_example()
        gaussian_filtering_example()
        batch_rendering_example()
        px2gid_analysis_example()
        
        print("\\n" + "=" * 50)
        print("✓ 所有示例运行完成!")
        print("生成的文件:")
        print("  - output_basic.png: 基础渲染结果")
        print("  - output_depth.png: 深度图")
        print("  - output_filtered.png: 过滤后渲染结果")
        print("  - output_view_*.png: 多视角渲染结果")
        print("  - px2gid_heatmap.png: px2gid分布热图")
        
    except Exception as e:
        print(f"\\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()