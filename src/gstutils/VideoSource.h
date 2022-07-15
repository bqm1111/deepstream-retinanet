#ifndef VIDEO_SOURCE_H
#define VIDEO_SOURCE_H
#include <gst/gst.h>
#include "common.h"
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

class GstVideoSrc
{
public:
    GstVideoSrc(std::string name_) : name(name_) {}

    GstElement *source = NULL;
    GstElement *demux = NULL;
    GstElement *parser = NULL;
    GstElement *decoder = NULL;
    std::string name;
    void linkbasic(std::string video_path, int source_id, GstElement *pipeline);
    void add_source_to_muxer(std::string video_path, int source_id, GstElement *pipeline, GstElement *muxer);
    void add_source_to_sink(std::string video_path, int source_id, GstElement *pipeline);
};

#endif