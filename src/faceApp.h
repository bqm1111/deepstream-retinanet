#ifndef APP_H
#define APP_H
#include "PipelineHandler.h"
#include <gst/gstelement.h>
#include <gst/gstpipeline.h>
#include <string>
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
    void linkMuxer();
    void showVideo();
    void faceDetection();
    void detectAndSend();
    GstElement * getPipeline();
};
#endif