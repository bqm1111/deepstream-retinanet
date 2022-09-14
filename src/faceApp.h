#ifndef APP_H
#define APP_H
#include <gst/gstelement.h>
#include <gst/gstpipeline.h>
#include <string>
#include <gst/gstpad.h>

#include "PipelineHandler.h"
#include "face_bin.h"
#include "mot_bin.h"
class FaceApp
{
public:
    FaceApp(std::string name);
    ~FaceApp();

    std::vector<std::string> m_video_source_path;
    GstAppParam m_gstparam;
    AppPipeline m_pipeline;
    std::string m_pipeline_name;
    void add_video(std::string video_path, std::string video_name);
    void linkMuxer();
    void showVideo();
    void faceDetection();
    void MOT();
    void detectAndSend();
    GstElement * getPipeline();
};
#endif