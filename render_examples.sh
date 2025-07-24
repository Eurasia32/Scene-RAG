#!/bin/bash

# 3D高斯溅射渲染器使用示例脚本

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <PLY文件路径> [输出目录]"
    echo "示例: $0 model.ply ./output"
    exit 1
fi

PLY_FILE=$1
OUTPUT_DIR=${2:-"./render_output"}

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 渲染器可执行文件路径
RENDERER="./build/opensplat_render"

# 检查渲染器是否存在
if [ ! -f "$RENDERER" ]; then
    echo "错误: 找不到渲染器 $RENDERER"
    echo "请先构建项目: cd build && make"
    exit 1
fi

# 检查PLY文件是否存在
if [ ! -f "$PLY_FILE" ]; then
    echo "错误: PLY文件不存在: $PLY_FILE"
    exit 1
fi

echo "开始渲染 $PLY_FILE..."

# 示例1: 基本渲染 (正视图)
echo "渲染示例1: 正视图"
$RENDERER \
  -i "$PLY_FILE" \
  -o "$OUTPUT_DIR/front_view.png" \
  --width 800 --height 600 \
  -m "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"

# 示例2: 侧视图
echo "渲染示例2: 侧视图"
$RENDERER \
  -i "$PLY_FILE" \
  -o "$OUTPUT_DIR/side_view.png" \
  --width 800 --height 600 \
  -m "0.0 0.0 1.0 5.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0"

# 示例3: 俯视图
echo "渲染示例3: 俯视图"
$RENDERER \
  -i "$PLY_FILE" \
  -o "$OUTPUT_DIR/top_view.png" \
  --width 800 --height 600 \
  -m "1.0 0.0 0.0 0.0 0.0 0.0 -1.0 5.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0"

# 示例4: 高分辨率渲染
echo "渲染示例4: 高分辨率"
$RENDERER \
  -i "$PLY_FILE" \
  -o "$OUTPUT_DIR/high_res.png" \
  --width 1920 --height 1080 \
  -m "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"

# 示例5: 降采样快速预览
echo "渲染示例5: 降采样预览"
$RENDERER \
  -i "$PLY_FILE" \
  -o "$OUTPUT_DIR/preview.png" \
  -d 4.0 \
  --width 1600 --height 1200 \
  -m "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"

# 示例6: 不同球谐函数阶数
echo "渲染示例6: 低阶球谐函数"
$RENDERER \
  -i "$PLY_FILE" \
  -o "$OUTPUT_DIR/low_sh.png" \
  -s 1 \
  --width 800 --height 600 \
  -m "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 5.0 0.0 0.0 0.0 1.0"

echo "渲染完成！输出文件保存在 $OUTPUT_DIR/"
echo "生成的文件:"
ls -la "$OUTPUT_DIR/"*.png