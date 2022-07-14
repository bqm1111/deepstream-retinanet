#ifndef VIDEO_SOURCE_H
#define VIDEO_SOURCE_H
#include <gst/gst.h>
#include "utils.h"
#include <gst/gstelement.h>
#include <gst/gstpad.h>
#include <iostream>
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

class VideoSource
{
public:
    VideoSource(){};

    GstElement *source = NULL;
    GstElement *demux = NULL;
    GstElement *parser = NULL;
    GstElement *decoder = NULL;

    static void newPadCB(GstElement * element, GstPad *pad, gpointer data);
    void add_source(std::string video_path, int source_id, GstElement * pipeline, GstElement * muxer);
};

#endif