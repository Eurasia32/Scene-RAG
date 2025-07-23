#ifndef CAM_H
#define CAM_H

#include <fstream>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <unordered_map>

struct Cam {
  int width = 0;
  int height = 0;
  float fx = 0;
  float fy = 0;
  torch::Tensor camToWorld;

  Cam() {};
  Cam(int width, int height, float fx, float fy,
      const torch::Tensor &camToWorld)
      : width(width), height(height), fx(fx), fy(fy), camToWorld(camToWorld) {}
};

#endif