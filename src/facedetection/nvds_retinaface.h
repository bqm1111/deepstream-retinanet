#ifndef _NVDS_RETINAFACE_H_
#define _NVDS_RETINAFACE_H_
// This file is somewhat similar to Retinaface.h
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"

#include <iostream>
#include <map>

#ifndef RETINAFACE_WEIGHT_PATH
#define RETINAFACE_WEIGHT_PATH "../data/models/wts/retinaface-r50.wts"
#endif

using namespace nvinfer1;

class RetinafaceDS : public IModelParser
{
public:
    RetinafaceDS(int net_H, int net_W, int32_t maxBatchSize, nvinfer1::DataType inputType, std::string weighFile);
    ~RetinafaceDS();
    bool hasFullDimsSupported() const override { return false; }
    const char *getModelName() const override
    {
        return "retinaface_r50";
    }

    NvDsInferStatus parseModel(nvinfer1::INetworkDefinition &network) override;

public:
    int m_input_h, m_input_w, m_output_size;
private:
    nvinfer1::DataType m_inputType;
    int m_maxBatchSize;
    std::map<std::string, nvinfer1::Weights> m_weightMap;
    std::string m_weight_file;
};

extern "C" bool NvDsInferRetinafaceCudaEngineGet(nvinfer1::IBuilder *const builder,
                                                 nvinfer1::IBuilderConfig *const builderConfig,
                                                 const NvDsInferContextInitParams *const initParams,
                                                 nvinfer1::DataType dataType,
                                                 nvinfer1::ICudaEngine *&cudaEngine);

CHECK_CUSTOM_ENGINE_CREATE_FUNC_PROTOTYPE(NvDsInferRetinafaceCudaEngineGet);

/**
 * @brief This function can only parse bbox. Useful when use gst-launch-1.0
 * 
 */
extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);


CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomRetinaface);

/**
 * @brief when output raw tensor, bbox + facemask parsed by pgie_src_pad_buffer_probe
 * This function do nothing
 * 
 */
extern "C" bool NvDsInferParseNone(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseNone);

#endif // _NVDS_RETINAFACE_H_