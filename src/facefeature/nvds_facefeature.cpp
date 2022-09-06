// #ifndef _NVDS_FACEFEATURE_H_
// #define _NVDS_FACEFEATURE_H_

#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"

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

// CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseNone);

// #endif // _NVDS_FACEFEATURE_H_
