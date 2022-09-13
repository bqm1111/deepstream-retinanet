#include "faceNMSPlugin.h"
#include <NvInferPluginUtils.h>
#include <NvInferRuntimeCommon.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::FaceNMSBasePluginCreator;
using nvinfer1::plugin::FaceNMSDynamicPlugin;
using nvinfer1::plugin::FaceNMSDynamicPluginCreator;
using nvinfer1::plugin::NMSParams;

namespace
{
    const char *FACE_NMS_PLUGIN_VERSION{"1"};
    const char *FACE_NMS_PLUGIN_NAME[] = {"FaceNMSDynamic_TRT"};
}

namespace nvinfer1
{
    namespace plugin
    {
        template <>
        void write<NMSParams>(char *&buffer, const NMSParams &val)
        {
            auto param = reinterpret_cast<NMSParams *>(buffer);
            std::memset(param, 0, sizeof(NMSParams));
            param->iou_threshold = val.iou_threshold;
            param->post_cluster_threshold = val.post_cluster_threshold;
            buffer += sizeof(NMSParams);
        }
    };
};

PluginFieldCollection FaceNMSBasePluginCreator::mFC{};
std::vector<PluginField> FaceNMSBasePluginCreator::mPluginAttributes;
static inline pluginStatus_t checkParams(const NMSParams &param)
{
    // NMS plugin supports maximum thread blocksize of 512 and upto 8 blocks at once.
    constexpr int32_t maxTopK{512 * 8};
    if (param.topK > maxTopK)
    {
        std::cout << "Invalid parameter: NMS topK (" << param.topK << ") exceeds limit (" << maxTopK << ")" << std::endl;
        return STATUS_BAD_PARAM;
    }

    return STATUS_SUCCESS;
}

FaceNMSDynamicPlugin::FaceNMSDynamicPlugin(NMSParams params) : param(params)
{
    mPluginStatus = checkParams(param);
}

FaceNMSDynamicPlugin::FaceNMSDynamicPlugin(const void *data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<NMSParams>(d);
    PLUGIN_VALIDATE(d == a + length);

    mPluginStatus = checkParams(param);
}

int FaceNMSDynamicPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int FaceNMSDynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void FaceNMSDynamicPlugin::terminate() noexcept {}

DimsExprs FaceNMSDynamicPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // try
    // {
    //     PLUGIN_ASSERT(nbInputs == 1);
    //     PLUGIN_ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
        
    // }
    // catch
    // {

    // }
}