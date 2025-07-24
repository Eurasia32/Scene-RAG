// 3D Gaussian Splatting 渲染核心 - 仅正向传播版本
// 基于 gsplat 项目，专为渲染优化
// Licensed under the AGPLv3

#include "bindings_render.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

using namespace torch::indexing;

// 基础数学结构体
struct Vector2 {
  float x, y;
};
struct Vector3 {
  float x, y, z;
};
struct Vector4 {
  float x, y, z, w;
};
struct Matrix2x2 {
  float data[2][2];
};
struct Matrix3x3 {
  float data[3][3];
};
struct Matrix4x4 {
  float data[4][4];
};

// 矩阵运算
Matrix3x3 multiply(const Matrix3x3 &a, const Matrix3x3 &b) {
  Matrix3x3 res{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        res.data[i][j] += a.data[i][k] * b.data[k][j];
  return res;
}

Matrix3x3 transpose(const Matrix3x3 &a) {
  return {{{a.data[0][0], a.data[1][0], a.data[2][0]},
           {a.data[0][1], a.data[1][1], a.data[2][1]},
           {a.data[0][2], a.data[1][2], a.data[2][2]}}};
}

// 四元数转旋转矩阵
Matrix3x3 quat_to_rot(const Vector4 &q) {
  const float w = q.w, x = q.x, y = q.y, z = q.z;
  return {{{1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w},
           {2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w},
           {2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y}}};
}

// 投影高斯到2D屏幕空间
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
project_gaussians_forward_tensor_cpu(
    const int num_points, torch::Tensor &means3d, torch::Tensor &scales,
    const float glob_scale, torch::Tensor &quats, torch::Tensor &viewmat,
    torch::Tensor &projmat, const float fx, const float fy, const float cx,
    const float cy, const unsigned img_height, const unsigned img_width,
    const float clip_thresh) {

  float eps = 1e-6f;
  float fovx = 0.5f * static_cast<float>(img_width) / fx;
  float fovy = 0.5f * static_cast<float>(img_height) / fy;
  float limX = 1.3f * fovx;
  float limY = 1.3f * fovy;

  // 获取数据指针
  float *viewmat_ptr = viewmat.data_ptr<float>();
  float *means3d_ptr = means3d.data_ptr<float>();
  float *quats_ptr = quats.data_ptr<float>();
  float *scale_ptr = scales.data_ptr<float>();
  float *projmat_ptr = projmat.data_ptr<float>();

  // 提取视图变换矩阵
  const Matrix3x3 Rclip = {{{viewmat_ptr[0], viewmat_ptr[1], viewmat_ptr[2]},
                            {viewmat_ptr[4], viewmat_ptr[5], viewmat_ptr[6]},
                            {viewmat_ptr[8], viewmat_ptr[9], viewmat_ptr[10]}}};
  const Vector3 Tclip = {viewmat_ptr[3], viewmat_ptr[7], viewmat_ptr[11]};

  // 输出数据向量
  std::vector<float> v_xys, v_conic, v_cov2d, v_camDepths;
  std::vector<int> v_radii;

  for (int i = 0; i < num_points; i++) {
    // 获取3D点位置
    Vector3 p = {means3d_ptr[3 * i], means3d_ptr[3 * i + 1],
                 means3d_ptr[3 * i + 2]};

    // 变换到相机空间
    Vector3 p_view = {Rclip.data[0][0] * p.x + Rclip.data[0][1] * p.y +
                          Rclip.data[0][2] * p.z + Tclip.x,
                      Rclip.data[1][0] * p.x + Rclip.data[1][1] * p.y +
                          Rclip.data[1][2] * p.z + Tclip.y,
                      Rclip.data[2][0] * p.x + Rclip.data[2][1] * p.y +
                          Rclip.data[2][2] * p.z + Tclip.z};

    // 获取四元数和缩放
    Vector4 quat = {quats_ptr[4 * i], quats_ptr[4 * i + 1],
                    quats_ptr[4 * i + 2], quats_ptr[4 * i + 3]};
    Vector3 scale = {scale_ptr[3 * i], scale_ptr[3 * i + 1],
                     scale_ptr[3 * i + 2]};

    // 计算3D协方差矩阵
    Matrix3x3 R = quat_to_rot(quat);
    Matrix3x3 S = {{{scale.x * glob_scale, 0, 0},
                    {0, scale.y * glob_scale, 0},
                    {0, 0, scale.z * glob_scale}}};
    Matrix3x3 M = multiply(R, S);
    Matrix3x3 cov3d = multiply(M, transpose(M));

    // 计算投影参数
    float minLimX =
        p_view.z * std::min(limX, std::max(-limX, p_view.x / p_view.z));
    float minLimY =
        p_view.z * std::min(limY, std::max(-limY, p_view.y / p_view.z));

    float rz = 1.0f / p_view.z;
    float rz2 = rz * rz;
    Matrix3x3 J = {{{fx * rz, 0, -fx * minLimX * rz2},
                    {0, fy * rz, -fy * minLimY * rz2},
                    {0, 0, 0}}};

    // 投影3D协方差到2D
    Matrix3x3 T = multiply(J, Rclip);
    Matrix3x3 cov2d = multiply(T, multiply(cov3d, transpose(T)));

    // 添加模糊以增强数值稳定性
    cov2d.data[0][0] += 0.3f;
    cov2d.data[1][1] += 0.3f;

    // 计算2D椭圆参数
    float a1 = cov2d.data[0][0], a2 = cov2d.data[1][1], a3 = cov2d.data[0][1];
    float det = std::max(a1 * a2 - a3 * a3, eps);
    Vector3 conic = {a2 / det, -a3 / det, a1 / det};

    // 计算椭圆半径
    float b = (a1 + a2) / 2.0f;
    float sq = std::sqrt(std::max(b * b - det, 0.1f));
    float v1 = b + sq, v2 = b - sq;
    float radius = std::ceil(3.0f * std::sqrt(std::max(v1, v2)));
    int radii = static_cast<int>(radius);

    // 投影到屏幕坐标
    Vector4 pHom = {p.x, p.y, p.z, 1.0f};
    pHom = {projmat_ptr[0] * pHom.x + projmat_ptr[1] * pHom.y +
                projmat_ptr[2] * pHom.z + projmat_ptr[3],
            projmat_ptr[4] * pHom.x + projmat_ptr[5] * pHom.y +
                projmat_ptr[6] * pHom.z + projmat_ptr[7],
            projmat_ptr[8] * pHom.x + projmat_ptr[9] * pHom.y +
                projmat_ptr[10] * pHom.z + projmat_ptr[11],
            projmat_ptr[12] * pHom.x + projmat_ptr[13] * pHom.y +
                projmat_ptr[14] * pHom.z + projmat_ptr[15]};

    float rw = 1.0f / std::max(pHom.w, eps);
    Vector3 pProj = {rw * pHom.x, rw * pHom.y, rw * pHom.z};
    float u = 0.5f * ((pProj.x + 1.0f) * static_cast<float>(img_width) - 1.0f);
    float v = 0.5f * ((pProj.y + 1.0f) * static_cast<float>(img_height) - 1.0f);

    // 存储结果
    v_xys.push_back(u);
    v_xys.push_back(v);
    v_radii.push_back(radii);
    v_conic.push_back(conic.x);
    v_conic.push_back(conic.y);
    v_conic.push_back(conic.z);
    v_cov2d.push_back(a1);
    v_cov2d.push_back(a3);
    v_cov2d.push_back(a3);
    v_cov2d.push_back(a2);
    v_camDepths.push_back(pProj.z);
  }

  // 转换为PyTorch张量
  torch::Tensor xys = torch::from_blob(v_xys.data(), {num_points, 2});
  torch::Tensor radii =
      torch::from_blob(v_radii.data(), {num_points}, torch::kInt32);
  torch::Tensor conic = torch::from_blob(v_conic.data(), {num_points, 3});
  torch::Tensor cov2d = torch::from_blob(v_cov2d.data(), {num_points, 2, 2});
  torch::Tensor camDepths =
      torch::from_blob(v_camDepths.data(), {num_points}, torch::kFloat32);

  return std::make_tuple(xys.clone(), radii.clone(), conic.clone(),
                         cov2d.clone(), camDepths.clone());
}

// 2D光栅化 - Alpha混合
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<int32_t> *>
rasterize_forward_tensor_cpu(
    const int width, const int height, const torch::Tensor &xys,
    const torch::Tensor &conics, const torch::Tensor &colors,
    const torch::Tensor &opacities, const torch::Tensor &background,
    const torch::Tensor &cov2d, const torch::Tensor &camDepths) {

  torch::NoGradGuard noGrad;

  int channels = colors.size(1);
  int numPoints = xys.size(0);
  float *pDepths = static_cast<float *>(camDepths.data_ptr());
  std::vector<int32_t> *px2gid = new std::vector<int32_t>[width * height];

  // 按深度排序高斯
  std::vector<size_t> gIndices(numPoints);
  std::iota(gIndices.begin(), gIndices.end(), 0);
  std::sort(gIndices.begin(), gIndices.end(),
            [&pDepths](int a, int b) { return pDepths[a] < pDepths[b]; });

  torch::Device device = xys.device();

  // 初始化输出张量
  torch::Tensor outImg = torch::zeros(
      {height, width, channels},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor outDepth = torch::zeros(
      {height, width},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor finalTs =
      torch::ones({height, width},
                  torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor done =
      torch::zeros({height, width},
                   torch::TensorOptions().dtype(torch::kBool).device(device));

  // 计算椭圆边界
  torch::Tensor sqCov2dX = 3.0f * torch::sqrt(cov2d.index({"...", 0, 0}));
  torch::Tensor sqCov2dY = 3.0f * torch::sqrt(cov2d.index({"...", 1, 1}));

  // 获取数据指针
  float *pConics = static_cast<float *>(conics.data_ptr());
  float *pCenters = static_cast<float *>(xys.data_ptr());
  float *pSqCov2dX = static_cast<float *>(sqCov2dX.data_ptr());
  float *pSqCov2dY = static_cast<float *>(sqCov2dY.data_ptr());
  float *pOpacities = static_cast<float *>(opacities.data_ptr());
  
  float *pOutImg = static_cast<float *>(outImg.data_ptr());
  float *pOutDepth = static_cast<float *>(outDepth.data_ptr());
  float *pFinalTs = static_cast<float *>(finalTs.data_ptr());
  bool *pDone = static_cast<bool *>(done.data_ptr());
  float *pColors = static_cast<float *>(colors.data_ptr());

  float bgR = background[0].item<float>();
  float bgG = background[1].item<float>();
  float bgB = background[2].item<float>();

  const float alphaThresh = 1.0f / 255.0f;

  // 逐个渲染高斯
  for (int idx = 0; idx < numPoints; idx++) {
    int32_t gaussianId = gIndices[idx];

    float A = pConics[gaussianId * 3 + 0];
    float B = pConics[gaussianId * 3 + 1];
    float C = pConics[gaussianId * 3 + 2];

    float gX = pCenters[gaussianId * 2 + 0];
    float gY = pCenters[gaussianId * 2 + 1];
    float gDepth = pDepths[gaussianId]; // 获取该高斯的深度值

    float sqx = pSqCov2dX[gaussianId];
    float sqy = pSqCov2dY[gaussianId];

    // 计算包围盒
    int minx = std::max(0, static_cast<int>(std::floor(gY - sqy)) - 2);
    int maxx = std::min(height, static_cast<int>(std::ceil(gY + sqy)) + 2);
    int miny = std::max(0, static_cast<int>(std::floor(gX - sqx)) - 2);
    int maxy = std::min(width, static_cast<int>(std::ceil(gX + sqx)) + 2);

    // 逐像素渲染
    for (int i = minx; i < maxx; i++) {
      for (int j = miny; j < maxy; j++) {
        size_t pixIdx = (i * width + j);
        if (pDone[pixIdx])
          continue;

        float xCam = gX - j;
        float yCam = gY - i;
        float sigma =
            0.5f * (A * xCam * xCam + C * yCam * yCam) + B * xCam * yCam;

        if (sigma < 0.0f)
          continue;
        float alpha =
            std::min(0.999f, pOpacities[gaussianId] * std::exp(-sigma));
        if (alpha < alphaThresh)
          continue;

        float T = pFinalTs[pixIdx];
        float nextT = T * (1.0f - alpha);
        if (nextT <= 1e-4f) {
          pDone[pixIdx] = true;
          continue;
        }

        float vis = alpha * T;

        // Alpha混合RGB和深度
        pOutImg[pixIdx * 3 + 0] += vis * pColors[gaussianId * 3 + 0];
        pOutImg[pixIdx * 3 + 1] += vis * pColors[gaussianId * 3 + 1];
        pOutImg[pixIdx * 3 + 2] += vis * pColors[gaussianId * 3 + 2];
        
        // Alpha混合深度
        pOutDepth[pixIdx] += vis * gDepth;

        pFinalTs[pixIdx] = nextT;
        px2gid[pixIdx].push_back(gaussianId);
      }
    }
  }

  // 添加背景色和背景深度
  const float bgDepth = 1000.0f; // 背景深度设为远平面
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      size_t pixIdx = (i * width + j);
      float T = pFinalTs[pixIdx];

      // 背景RGB
      pOutImg[pixIdx * 3 + 0] += T * bgR;
      pOutImg[pixIdx * 3 + 1] += T * bgG;
      pOutImg[pixIdx * 3 + 2] += T * bgB;
      
      // 背景深度
      pOutDepth[pixIdx] += T * bgDepth;

      std::reverse(px2gid[pixIdx].begin(), px2gid[pixIdx].end());
    }
  }

  return std::make_tuple(outImg, outDepth, finalTs, px2gid);
}

// 球谐函数常数
const float SH_C0 = 0.28209479177387814f;
const float SH_C1 = 0.4886025119029199f;
const float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f,
                       0.31539156525252005f, -1.0925484305920792f,
                       0.5462742152960396f};
const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f,
                       -0.4570457994644658f, 0.3731763325901154f,
                       -0.4570457994644658f, 1.445305721320277f,
                       -0.5900435899266435f};
const float SH_C4[] = {
    2.5033429417967046f,  -1.7701307697799304f, 0.9461746957575601f,
    -0.6690465435572892f, 0.10578554691520431f, -0.6690465435572892f,
    0.47308734787878004f, -1.7701307697799304f, 0.6258357354491761f};

int numShBases(int degree) {
  switch (degree) {
  case 0:
    return 1;
  case 1:
    return 4;
  case 2:
    return 9;
  case 3:
    return 16;
  default:
    return 25;
  }
}

// 球谐函数正向计算
torch::Tensor compute_sh_forward_tensor_cpu(const int degrees_to_use,
                                            const torch::Tensor &viewdirs,
                                            const torch::Tensor &coeffs) {
  unsigned numBases = numShBases(degrees_to_use);

  torch::Tensor result = torch::zeros(
      {viewdirs.size(0), coeffs.size(-2)},
      torch::TensorOptions().dtype(torch::kFloat32).device(viewdirs.device()));

  result.index_put_({"...", 0}, SH_C0);

  if (numBases > 1) {
    std::vector<torch::Tensor> xyz = viewdirs.unbind(-1);
    torch::Tensor x = xyz[0], y = xyz[1], z = xyz[2];

    result.index_put_({"...", 1}, SH_C1 * -y);
    result.index_put_({"...", 2}, SH_C1 * z);
    result.index_put_({"...", 3}, SH_C1 * -x);

    if (numBases > 4) {
      torch::Tensor xx = x * x, yy = y * y, zz = z * z;
      torch::Tensor xy = x * y, yz = y * z, xz = x * z;

      result.index_put_({"...", 4}, SH_C2[0] * xy);
      result.index_put_({"...", 5}, SH_C2[1] * yz);
      result.index_put_({"...", 6}, SH_C2[2] * (2.0f * zz - xx - yy));
      result.index_put_({"...", 7}, SH_C2[3] * xz);
      result.index_put_({"...", 8}, SH_C2[4] * (xx - yy));

      if (numBases > 9) {
        result.index_put_({"...", 9}, SH_C3[0] * y * (3 * xx - yy));
        result.index_put_({"...", 10}, SH_C3[1] * xy * z);
        result.index_put_({"...", 11}, SH_C3[2] * y * (4 * zz - xx - yy));
        result.index_put_({"...", 12},
                          SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy));
        result.index_put_({"...", 13}, SH_C3[4] * x * (4 * zz - xx - yy));
        result.index_put_({"...", 14}, SH_C3[5] * z * (xx - yy));
        result.index_put_({"...", 15}, SH_C3[6] * x * (xx - 3 * yy));

        if (numBases > 16) {
          result.index_put_({"...", 16}, SH_C4[0] * xy * (xx - yy));
          result.index_put_({"...", 17}, SH_C4[1] * yz * (3 * xx - yy));
          result.index_put_({"...", 18}, SH_C4[2] * xy * (7 * zz - 1));
          result.index_put_({"...", 19}, SH_C4[3] * yz * (7 * zz - 3));
          result.index_put_({"...", 20}, SH_C4[4] * (zz * (35 * zz - 30) + 3));
          result.index_put_({"...", 21}, SH_C4[5] * xz * (7 * zz - 3));
          result.index_put_({"...", 22}, SH_C4[6] * (xx - yy) * (7 * zz - 1));
          result.index_put_({"...", 23}, SH_C4[7] * xz * (xx - 3 * yy));
          result.index_put_({"...", 24}, SH_C4[8] * (xx * (xx - 3 * yy) -
                                                     yy * (3 * xx - yy)));
        }
      }
    }
  }

  return (result.index({"...", None}) * coeffs).sum(-2);
}