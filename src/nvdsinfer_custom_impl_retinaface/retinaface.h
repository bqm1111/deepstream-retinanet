#ifndef RETINA_FACE_H_a33c9dab6f0f0043ae5781d0
#define RETINA_FACE_H_a33c9dab6f0f0043ae5781d0
#include "NvInferPlugin.h"
#include <NvInfer.h>
#include <vector>
#include "cuda_runtime_api.h"
#include <cassert>
#include <cublas_v2.h>
#include <functional>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"

using namespace nvinfer1;
// 
class RetinaFace : public IModelParser
{
public:
    RetinaFace(int net_H, int net_W, int32_t maxBatchSize, nvinfer1::DataType inputType, std::string weightFile);
    ~RetinaFace();

    bool hasFullDimsSupported() const override { return false; }

    const char *getModelName() const override
    {
        return "retinaface_r50";
    }

    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition& network) override;

    // nvinfer1::ICudaEngine *createEngine (nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config);
};
#endif