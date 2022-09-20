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
#include "params.h"

class NvInferBinBase
{
public:
    NvInferBinBase(){};
    NvInferBinBase(std::shared_ptr<NvInferBinConfigBase> configs) : m_configs(configs){};

    virtual ~NvInferBinBase() {}
    void getMasterBin(GstElement *&bin) { bin = this->m_masterBin; }
    void setParam(GstAppParam param) { m_params = param; };
    virtual void createInferBin() = 0;
    virtual void attachProbe() = 0;
    virtual void setMsgBrokerConfig() = 0;

    GstElement *createInferPipeline(GstElement *pipeline);
    void createVideoSinkBin();
    void createFileSinkBin(std::string location);
    void linkMsgBroker();

    GstElement *m_pipeline = NULL;
    // Common element in infer bin
    GstElement *m_pgie = NULL;
    GstElement *m_sgie = NULL;

    // common elements for the rest of the pipeline
    GstElement *m_tiler = NULL;
    GstElement *m_convert = NULL;
    GstElement *m_msgconv = NULL;
    GstElement *m_msgbroker = NULL;
    GstElement *m_tee = NULL;
    GstElement *m_queue_display = NULL;
    GstElement *m_queue_msg = NULL;
    GstElement *m_osd = NULL;
    GstElement *m_file_convert = NULL;
    GstElement *m_capsfilter = NULL;
    GstElement *m_nvv4l2h265enc = NULL;
    GstElement *m_h265parse = NULL;
    GstElement *m_file_muxer = NULL;
    GstElement *m_sink = NULL;

    GstPad *m_tee_msg_pad;
    GstPad *m_tee_display_pad;

protected:
    GstAppParam m_params;
    GstElement *m_masterBin = NULL;
    std::shared_ptr<NvInferBinConfigBase> m_configs;
    std::string m_module_name;
};

#endif