#ifndef APP_H
#define APP_H
#include <string>
#include "PipelineHandler.h"
#include "FaceBin.h"
#include "MOTBin.h"
#include <curl/curl.h>

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
    GstElement *getPipeline();

    MOTTrackerList *m_tracker_list;
    CURL *m_curl;

private:
    void init_curl();
    void free_curl();
};
#endif