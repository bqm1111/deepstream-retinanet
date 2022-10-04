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
#include "mot_struct.h"
#include "utils.h"
#include "QDTLog.h"
class NvInferBinBase
{
public:
    NvInferBinBase(){};
    NvInferBinBase(std::shared_ptr<NvInferBinConfigBase> configs) : m_configs(configs){};

    virtual ~NvInferBinBase() {}
    void getMasterBin(GstElement *&bin) { bin = this->m_masterBin; }
    void setParam(GstAppParam param) { m_params = param; };
    virtual void createInferBin() {}
    virtual void attachProbe();
    virtual void setMsgBrokerConfig();
    void acquireTrackerList(MOTTrackerList *tracker_list) { m_tracker_list = tracker_list; }

    GstElement *createInferPipeline(GstElement *pipeline);
    GstElement *createNonInferPipeline(GstElement *pipeline);
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
    GstElement *m_metadata_msgconv = NULL;
    GstElement *m_metadata_msgbroker = NULL;
    GstElement *m_visual_msgconv = NULL;
    GstElement *m_visual_msgbroker = NULL;

    GstElement *m_tee = NULL;
    GstElement *m_queue_display = NULL;
    GstElement *m_queue_metadata_msg = NULL;
    GstElement *m_queue_visual_msg = NULL;

    GstElement *m_osd = NULL;
    GstElement *m_file_convert = NULL;
    GstElement *m_capsfilter = NULL;
    GstElement *m_nvv4l2h265enc = NULL;
    GstElement *m_h265parse = NULL;
    GstElement *m_file_muxer = NULL;
    GstElement *m_sink = NULL;

    GstPad *m_tee_display_pad;
    GstPad *m_tee_metadata_msg_pad;
    GstPad *m_tee_visual_msg_pad;

    static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
    static GstPadProbeReturn tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);

protected:
    MOTTrackerList *m_tracker_list;
    GstAppParam m_params;
    GstElement *m_masterBin = NULL;
    std::shared_ptr<NvInferBinConfigBase> m_configs;
    std::string m_module_name;
};

#endif