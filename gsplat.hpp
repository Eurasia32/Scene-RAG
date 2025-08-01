#ifndef GSPLAT_H
#define GSPLAT_H

#include "./rasterizer/gsplat/config.h"

#if defined(USE_HIP) || defined(USE_CUDA)
#include "./rasterizer/gsplat/bindings.h"
#endif

#include "./rasterizer/gsplat-cpu/bindings.h"

#endif
