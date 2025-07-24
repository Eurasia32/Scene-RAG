#!/bin/bash

# 安全清理脚本 - 移除渲染专用工程不需要的文件
# 此脚本会将文件移动到backup目录而不是直接删除

echo "开始清理不必要的文件..."

# 创建备份目录
BACKUP_DIR="./backup_removed_files"
mkdir -p "$BACKUP_DIR/src"
mkdir -p "$BACKUP_DIR/include"

echo "创建备份目录: $BACKUP_DIR"

# 备份并移除不需要的源文件
echo "移除不需要的源文件..."
if [ -f "src/opensplat.cpp" ]; then
    mv src/opensplat.cpp "$BACKUP_DIR/src/"
    echo "  ✓ 移除 src/opensplat.cpp"
fi

if [ -f "src/model.cpp" ]; then
    mv src/model.cpp "$BACKUP_DIR/src/"
    echo "  ✓ 移除 src/model.cpp"
fi

if [ -f "src/cv_utils.cpp" ]; then
    mv src/cv_utils.cpp "$BACKUP_DIR/src/"
    echo "  ✓ 移除 src/cv_utils.cpp"
fi

if [ -f "src/point_io.cpp" ]; then
    mv src/point_io.cpp "$BACKUP_DIR/src/"
    echo "  ✓ 移除 src/point_io.cpp"
fi

# 备份并移除不需要的头文件
echo "移除不需要的头文件..."
if [ -f "include/model.hpp" ]; then
    mv include/model.hpp "$BACKUP_DIR/include/"
    echo "  ✓ 移除 include/model.hpp"
fi

if [ -f "include/cv_utils.hpp" ]; then
    mv include/cv_utils.hpp "$BACKUP_DIR/include/"
    echo "  ✓ 移除 include/cv_utils.hpp"
fi

if [ -f "include/point_io.hpp" ]; then
    mv include/point_io.hpp "$BACKUP_DIR/include/"
    echo "  ✓ 移除 include/point_io.hpp"
fi

if [ -f "include/opensplat.hpp" ]; then
    mv include/opensplat.hpp "$BACKUP_DIR/include/"
    echo "  ✓ 移除 include/opensplat.hpp"
fi

if [ -f "include/cam.hpp" ]; then
    mv include/cam.hpp "$BACKUP_DIR/include/"
    echo "  ✓ 移除 include/cam.hpp"
fi

# 备份一些可能不需要的文件
if [ -f "CMakeLists_render.txt" ]; then
    mv CMakeLists_render.txt "$BACKUP_DIR/"
    echo "  ✓ 移除 CMakeLists_render.txt (已集成到主CMakeLists.txt)"
fi

echo ""
echo "清理完成！"
echo "被移除的文件已备份到: $BACKUP_DIR"
echo ""
echo "如果需要恢复任何文件，可以从备份目录复制回来。"
echo "当前保留的核心文件："
echo "  - src/opensplat_render.cpp (主渲染程序)"
echo "  - src/model_render.cpp (简化模型)"
echo "  - src/cv_utils_render.cpp (图像处理)"
echo "  - include/model_render.hpp"
echo "  - include/cv_utils_render.hpp"
echo "  - include/constants.hpp"
echo "  - src/rasterizer/ (渲染核心)"
echo ""
echo "现在可以构建渲染器:"
echo "  mkdir -p build && cd build"
echo "  cmake .."
echo "  make -j8"