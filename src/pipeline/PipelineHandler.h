#ifndef PIPELINE_HANDLER_H
#define PIPELINE_HANDLER_H
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
#include "BufferProbe.h"
#include "helper_function.h"
#include "params.h"
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#ifndef FACEID_PGIE_CONFIG_PATH
// #define FACEID_PGIE_CONFIG_PATH "../configs/faceid/faceid_primary.txt"
#define FACEID_PGIE_CONFIG_PATH "../configs/config_infer_primary_yoloV5.txt"
#endif
#ifndef FACEID_ALIGN_CONFIG_PATH
#define FACEID_ALIGN_CONFIG_PATH "../configs/faceid/faceid_align_config.txt"
#endif
#ifndef FACEID_SGIE_CONFIG_PATH
#define FACEID_SGIE_CONFIG_PATH "../configs/faceid/faceid_secondary.txt"
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

    GstElement *m_muxer = NULL;
    GstElement *m_tiler = NULL;
    GstElement *m_convert = NULL;
    GstElement *m_msgconv = NULL;
    GstElement *m_msgbroker = NULL;
    GstElement *m_tee = NULL;
    GstElement *m_queue_display = NULL;
    GstElement *m_queue_msg = NULL;
    GstElement *m_osd = NULL;
    GstElement *m_sink = NULL;

    GstPad * m_tee_msg_pad;
    GstPad * m_tee_display_pad;

    std::string m_pipeline_name;
    GstAppParam m_gstparams;
    CloudParam m_cloudParams;
    void create(std::string pipeline_name, GstAppParam params);
    GstElement *add_video_source(std::string video_path, std::string video_name);
    void linkMuxer();
    GstElement *createGeneralSinkBin();
    void link(GstElement *in_elem, GstElement *out_elem);
    std::map<std::string, int> m_video_source;
    void linkMsgBroker();
    int numVideoSrc();
};

#endif