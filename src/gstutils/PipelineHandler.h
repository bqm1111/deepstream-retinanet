#ifndef PIPELINE_HANDLER_H
#define PIPELINE_HANDLER_H
#include <gst/gst.h>
#include "common.h"
#include <gst/gstelement.h>
#include <gst/gstpad.h>
#include <iostream>
#include <map>

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
    AppPipeline(){}
    AppPipeline(std::string pipeline_name, GstAppParam params);
    ~AppPipeline();
    GstElement *m_pipeline = NULL;
    std::vector<GstElement *> m_source;
    std::vector<GstElement *> m_demux ;
    std::vector<GstElement *> m_parser;
    std::vector<GstElement *> m_decoder;

    GstElement *m_muxer = NULL;
    GstElement *m_tiler = NULL;
    GstElement *m_osd = NULL;
    GstElement *m_sink = NULL;

    std::string m_pipeline_name;
    GstAppParam m_gstparams;
    void create(std::string pipeline_name, GstAppParam params);
    GstElement *add_video_source(std::string video_path, std::string video_name);
    void linkMuxer();
    static void wrapperAddNewPad(GstElement *element, GstPad *pad, gpointer data);
    GstElement *createGeneralSinkBin();
    void link(GstElement *in_elem, GstElement *out_elem);
    std::map<std::string, int> m_video_source;
    int numVideoSrc();
};

#endif