#!/bin/bash

# 清理渲染核心文件的脚本 - 移除冗余和反向传播相关的文件

echo "开始清理渲染核心文件..."

# 创建备份目录
BACKUP_DIR="./backup_rasterizer_files"
mkdir -p "$BACKUP_DIR/src/rasterizer/gsplat-cpu"
mkdir -p "$BACKUP_DIR/src"

echo "创建备份目录: $BACKUP_DIR"

# 备份并移除冗余的rasterizer文件
echo "清理rasterizer目录..."

if [ -f "src/rasterizer/gsplat-cpu/gsplat_cpu.cpp" ]; then
    mv src/rasterizer/gsplat-cpu/gsplat_cpu.cpp "$BACKUP_DIR/src/rasterizer/gsplat-cpu/"
    echo "  ✓ 移除 src/rasterizer/gsplat-cpu/gsplat_cpu.cpp (包含反向传播)"
fi

if [ -f "src/rasterizer/gsplat-cpu/bindings.h" ]; then
    mv src/rasterizer/gsplat-cpu/bindings.h "$BACKUP_DIR/src/rasterizer/gsplat-cpu/"
    echo "  ✓ 移除 src/rasterizer/gsplat-cpu/bindings.h (原始版本)"
fi

if [ -d "src/rasterizer/gsplat-cpu" ]; then
    rmdir src/rasterizer/gsplat-cpu 2>/dev/null
    echo "  ✓ 移除空目录 src/rasterizer/gsplat-cpu/"
fi

# 备份并移除独立的gsplat_cpu.cpp（可能是重复的）
if [ -f "src/gsplat_cpu.cpp" ]; then
    mv src/gsplat_cpu.cpp "$BACKUP_DIR/src/"
    echo "  ✓ 移除 src/gsplat_cpu.cpp (与新的gsplat_render.cpp重复)"
fi

echo ""
echo "清理完成！现在使用的文件结构："
echo "  ✓ src/rasterizer/gsplat_render.cpp - 纯正向传播渲染核心"
echo "  ✓ src/rasterizer/bindings_render.h - 简化的头文件声明"
echo "  ✓ 移除了所有反向传播/梯度计算代码"
echo "  ✓ 移除了冗余的实现文件"
echo ""
echo "备份文件保存在: $BACKUP_DIR"
echo ""
echo "现在可以重新构建项目:"
echo "  cd build && make clean && make -j8"