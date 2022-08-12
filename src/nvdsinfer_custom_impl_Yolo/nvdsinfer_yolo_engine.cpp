#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"
#include "yoloPlugins.h"
#include "yolo.h"

#include <algorithm>

#define USE_CUDA_ENGINE_GET_API 1

static bool getYoloNetworkInfo (NetworkInfo &networkInfo, const NvDsInferContextInitParams* initParams)
{
    std::string yoloCfg = initParams->customNetworkConfigFilePath;
    std::string yoloType;

    std::transform (yoloCfg.begin(), yoloCfg.end(), yoloCfg.begin(), [] (uint8_t c) {
        return std::tolower (c);});

    yoloType = yoloCfg.substr(0, yoloCfg.find(".cfg"));

    networkInfo.inputBlobName = "data";
    networkInfo.networkType = yoloType;
    networkInfo.configFilePath = initParams->customNetworkConfigFilePath;
    networkInfo.wtsFilePath = initParams->modelFilePath;
    networkInfo.int8CalibPath = initParams->int8CalibrationFilePath;
    networkInfo.deviceType = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.numDetectedClasses = initParams->numDetectedClasses;
    networkInfo.clusterMode = initParams->clusterMode;
    
    if(initParams->networkMode == 0) {
        networkInfo.networkMode = "FP32";
    }
    else if(initParams->networkMode == 1) {
        networkInfo.networkMode = "INT8";
    }
    else if(initParams->networkMode == 2) {
        networkInfo.networkMode = "FP16";
    }

    if (networkInfo.configFilePath.empty() ||
        networkInfo.wtsFilePath.empty()) {
        std::cerr << "YOLO config file or weights file is not specified\n" << std::endl;
        return false;
    }

    if (!fileExists(networkInfo.configFilePath) ||
        !fileExists(networkInfo.wtsFilePath)) {
        std::cerr << "YOLO config file or weights file is not exist\n" << std::endl;
        return false;
    }

    return true;
}

#if !USE_CUDA_ENGINE_GET_API
IModelParser* NvDsInferCreateModelParser(
    const NvDsInferContextInitParams* initParams) {
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
      return nullptr;
    }

    return new Yolo(networkInfo);
}
#else
extern "C"
bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder * const builder,
        nvinfer1::IBuilderConfig * const builderConfig,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder * const builder,
        nvinfer1::IBuilderConfig * const builderConfig,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{
    NetworkInfo networkInfo;
    if (!getYoloNetworkInfo(networkInfo, initParams)) {
      return false;
    }

    Yolo yolo(networkInfo);
    cudaEngine = yolo.createEngine (builder, builderConfig);
    if (cudaEngine == nullptr)
    {
        std::cerr << "Failed to build CUDA engine on "
                  << networkInfo.configFilePath << std::endl;
        return false;
    }

    return true;
}
#endif
