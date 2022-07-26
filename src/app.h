#ifndef APP_H
#define APP_H
#include "gstutils/VideoSource.h"
#include <gst/gstelement.h>
#include <string>
#include "common.h"
#include "custom_sink.hpp"

class FaceApp
{
public:
    FaceApp(std::string name);
    ~FaceApp();

    std::vector<std::string> m_video_source_path;
    GstAppParam m_gstparam;
    AppPipeline m_app;
    std::string m_app_name;
    void add_video(std::string video_path, std::string video_name);
    void showVideo();
};
#endif