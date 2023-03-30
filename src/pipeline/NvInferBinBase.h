#ifndef NVINFER_BIN_BASE_H
#define NVINFER_BIN_BASE_H
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/app/gstappsink.h>
#include <gst/gstobject.h>
#include <memory>
#include <nvdsinfer.h>
#include "NvInferBinConfigBase.h"
#include "common.h"
#include "params.h"
#include "app_struct.h"
#include "utils.h"
#include "QDTLog.h"

class NvInferBinBase
{
public:
    NvInferBinBase(){};
    NvInferBinBase(std::shared_ptr<NvInferBinConfigBase> configs) : m_configs(configs){};

    virtual ~NvInferBinBase() {}
    void getMasterBin(GstElement *&bin) { bin = this->m_masterBin; }
    virtual void createInferBin() {}
    virtual void attachProbe();
    void acquireUserData(user_callback_data *callback_data) { m_user_callback_data = callback_data; }

    GstElement *createInferPipeline(GstElement *pipeline);
    GstElement *createNonInferPipeline(GstElement *pipeline);
    void createVideoSinkBin();
    void createFileSinkBin(std::string location);
    void createAppSinkBin();

    GstElement *m_pipeline = NULL;
    // Common element in infer bin
    GstElement *m_pgie = NULL;
    GstElement *m_sgie = NULL;

    // common elements for the rest of the pipeline
    GstElement *m_tiler = NULL;
    GstElement *m_convert = NULL;

    GstElement *m_osd = NULL;
    GstElement *m_file_convert = NULL;
    GstElement *m_capsfilter = NULL;
    GstElement *m_nvv4l2h265enc = NULL;
    GstElement *m_h265parse = NULL;
    GstElement *m_file_muxer = NULL;
    GstElement *m_queue = NULL;
    GstElement *m_sink = NULL;

    static GstPadProbeReturn timer_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
    static GstPadProbeReturn tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
    static GstFlowReturn newSampleCallback(GstElement *sink, gpointer *user_data);


protected:
    user_callback_data *m_user_callback_data;
    GstElement *m_masterBin = NULL;
    std::shared_ptr<NvInferBinConfigBase> m_configs;
    std::string m_module_name;
};

#endif