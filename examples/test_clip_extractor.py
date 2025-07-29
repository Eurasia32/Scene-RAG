#!/usr/bin/env python3
"""
测试CLIP特征提取器实现
"""

import sys
import os
import numpy as np

# 添加python目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

def test_clip_extractor():
    """测试CLIP特征提取器"""
    print("开始测试CLIP特征提取器...")
    
    try:
        from intelligent_rag import IntelligentRAG
        
        # 创建模拟的模型路径
        model_path = "./test_model.ply"
        
        # 测试初始化（这里可能会因为没有真实的PLY文件而失败，但我们主要测试CLIP部分）
        print("1. 测试CLIP特征提取器初始化...")
        
        # 直接测试CLIP提取器类
        rag = IntelligentRAG.__new__(IntelligentRAG)  # 创建实例但不调用__init__
        
        # 直接测试CLIP初始化方法
        clip_extractor = rag._initialize_clip_extractor()
        print(f"✓ CLIP提取器初始化成功，设备: {clip_extractor.device}")
        
        # 测试文本特征提取
        print("2. 测试文本特征提取...")
        text_features = clip_extractor.extract_from_text("red chair")
        print(f"✓ 文本特征提取成功，维度: {text_features.shape}, 类型: {text_features.dtype}")
        print(f"   特征向量范围: [{text_features.min():.4f}, {text_features.max():.4f}]")
        print(f"   特征向量模长: {np.linalg.norm(text_features):.4f}")
        
        # 测试图像特征提取（创建一个简单的测试图像）
        print("3. 测试图像特征提取...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_features = clip_extractor.extract_from_image(test_image)
        print(f"✓ 图像特征提取成功，维度: {image_features.shape}, 类型: {image_features.dtype}")
        print(f"   特征向量范围: [{image_features.min():.4f}, {image_features.max():.4f}]")
        print(f"   特征向量模长: {np.linalg.norm(image_features):.4f}")
        
        # 测试特征相似度
        print("4. 测试特征相似度计算...")
        similarity = np.dot(text_features, image_features)
        print(f"✓ 文本-图像相似度: {similarity:.4f}")
        
        print("\n🎉 CLIP特征提取器测试完成！所有功能正常。")
        return True
        
    except ImportError as e:
        print(f"❌ 依赖导入失败: {e}")
        print("请安装必要依赖: pip install clip-by-openai pillow torch torchvision")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clip_extractor()
    sys.exit(0 if success else 1)