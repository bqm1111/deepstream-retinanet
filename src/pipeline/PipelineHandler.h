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
    ~AppPipeline();
    GstElement *m_pipeline = NULL;
    void setLiveSource(bool is_live);
    std::vector<GstElement *> m_source;
    std::vector<GstElement *> m_demux;
    std::vector<GstElement *> m_parser;
    std::vector<GstElement *> m_decoder;

    GstElement *m_stream_muxer = NULL;
    GstElement *m_tee_app = NULL;
    GstElement *m_queue_mot = NULL;
    GstElement *m_queue_face = NULL;
    std::string m_pipeline_name;

    void create(std::string pipeline_name);
    void add_video_source(std::vector<std::vector<std::string>> video_info, std::vector<std::string> video_name);
    void link(GstElement *in_elem, GstElement *out_elem);
    void linkMuxer(int muxer_output_width, int muxer_output_height);
    void linkTwoBranch(GstElement *mot_bin, GstElement *face_bin);
    std::unordered_map<std::string, int> m_video_source;
    bool m_live_source;
    int numVideoSrc();
};

#endif