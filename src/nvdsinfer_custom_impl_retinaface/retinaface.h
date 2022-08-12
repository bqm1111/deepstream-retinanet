#ifndef RETINA_FACE_H_a33c9dab6f0f0043ae5781d0
#define RETINA_FACE_H_a33c9dab6f0f0043ae5781d0
#include "NvInferPlugin.h"
#include <vector>
#include "cuda_runtime_api.h"
#include <cassert>
#include <cublas_v2.h>
#include <functional>
#include <numeric>
#include <algorithm>
#include <iostream>

using namespace nvinfer1;

class RetinaFace : public IPluginV2
{
public:
    RetinaFace();
};

#endif