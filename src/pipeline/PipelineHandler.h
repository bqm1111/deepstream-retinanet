#ifndef PIPELINE_HANDLER_H
#define PIPELINE_HANDLER_H
#include <gstreamer-1.0/gst/gstpad.h>
#include <iostream>
#include <string>
#include <map>
#include <c++/7/experimental/bits/fs_path.h>
#include <gst/gst.h>
#include <gst/gstpad.h>
#include <gst/gstcaps.h>
#include <gst/gstelementfactory.h>
#include <gst/gstinfo.h>
#include <gst/gstutils.h>
#include <gst/gstelement.h>
#include <opencv2/videostab/log.hpp>
#include "common.h"
#include "params.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

class AppPipeline
{
public:
    AppPipeline() {}
    AppPipeline(std::string pipeline_name, GstAppParam params);
    ~AppPipeline();
    GstElement *m_pipeline = NULL;
    std::vector<GstElement *> m_source;
    std::vector<GstElement *> m_demux;
    std::vector<GstElement *> m_parser;
    std::vector<GstElement *> m_decoder;
    
    GstElement *m_stream_muxer = NULL;
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

    std::string m_pipeline_name;
    GstAppParam m_gstparams;
    CloudParam m_cloudParams;
    void create(std::string pipeline_name, GstAppParam params);
    GstElement *add_video_source(std::string video_path, std::string video_name);
    void linkMuxer();
    void attachOsdProbe(GstPadProbeCallback callback);
    void attachTileProbe(GstPadProbeCallback callback);
    GstElement *createVideoSinkBin();
    GstElement *createFileSinkBin(std::string location);
    void link(GstElement *in_elem, GstElement *out_elem);
    std::map<std::string, int> m_video_source;
    void linkMsgBroker();
    int numVideoSrc();
};

#endif