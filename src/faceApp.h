#ifndef APP_H
#define APP_H
#include <string>
#include "PipelineHandler.h"
#include "FaceBin.h"
#include "MOTBin.h"
#include <curl/curl.h>
#include "utils.h"
#include "ConfigManager.h"
#include "DeepStreamAppConfig.h"
#include <uuid.h>

class FaceApp
{
public:
    FaceApp();
    ~FaceApp();

    void create(std::string name);
    void setLive(bool is_live);
    void loadConfig(std::string config_file);
    void addVideoSource(std::string list_video_src_file);
    void MOT();
    void detect();
    void detectAndMOT();
    void sequentialDetectAndMOT();
    GstElement *getPipeline();
    int numVideoSrc();
    std::vector<std::string> m_video_source_name;
    std::vector<std::vector<std::string>> m_video_source_info;
    GstAppParam m_gstparam;
    AppPipeline m_pipeline;
    MOTTrackerList *m_tracker_list = nullptr;

private:
    ConfigManager *m_config;
    user_callback_data *m_user_callback_data;
    void init_curl();
    void free_curl();
};
#endif