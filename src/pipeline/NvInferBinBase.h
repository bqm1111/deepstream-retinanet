#ifndef NVINFER_BIN_BASE_H
#define NVINFER_BIN_BASE_H
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstobject.h>
#include <memory>
#include <nvdsinfer.h>
#include "NvInferBinConfigBase.h"
#include "common.h"

class NvInferBinBase
{
public:
    NvInferBinBase(){};
    NvInferBinBase(std::shared_ptr<NvInferBinConfigBase> configs) : m_configs(configs){};

    virtual ~NvInferBinBase() {}
    void getMasterBin(GstElement *&bin) { bin = this->m_masterBin; }
    GstElement *pgie = NULL;
    GstElement *sgie = NULL;
    virtual void createBin() {}

protected:
    GstElement *m_masterBin = NULL;
    std::shared_ptr<NvInferBinConfigBase> m_configs;
};

#endif