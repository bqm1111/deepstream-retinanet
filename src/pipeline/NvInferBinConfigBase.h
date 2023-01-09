#ifndef NVINFER_BIN_CONFIG_BASE_H
#define NVINFER_BIN_CONFIG_BASE_H
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstobject.h>
#include <string>

class NvInferBinConfigBase
{
public:
    NvInferBinConfigBase() {}
    virtual ~NvInferBinConfigBase(){}
    NvInferBinConfigBase(std::string pgie, std::string sgie) : pgie_config_path(pgie),
                                                               sgie_config_path(sgie) {}
    std::string pgie_config_path;
    std::string sgie_config_path;
};

#endif