#ifndef APP_H
#define APP_H
#include "gstutils/VideoSource.h"
#include <gst/gstelement.h>
#include <string>
#include "common.h"


class FaceApp
{
public:
    FaceApp();
    ~FaceApp();

    std::vector<std::string> m_video_source_path;
    std::vector<GstVideoSrc> m_gst_vidsrc;
    GstAppParam m_gstparam;
    GstElement * m_pipeline;
    GstElement * m_muxer;
    void add_video_source(std::string video_path, std::string name);
    void linkMultipleSrcToMuxer();
    void linkToShowVideo();
    int getNumVideoSource();
    void executePipeline();
};
#endif