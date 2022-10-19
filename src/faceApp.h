#ifndef APP_H
#define APP_H
#include <string>
#include "FaceBin.h"
#include "MOTBin.h"
#include <curl/curl.h>
#include "utils.h"
#include "ConfigManager.h"
#include "DeepStreamAppConfig.h"
#include <uuid.h>
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
class FaceApp
{
public:
    FaceApp(std::string app_name);
    ~FaceApp();

    void loadConfig();
    void addVideoSource(std::string list_video_src_file);
    void sequentialDetectAndMOT();
    GstElement *getPipeline();
    int numVideoSrc();
    std::vector<std::string> m_video_source_name;
    std::vector<std::vector<std::string>> m_video_source_info;
    GstElement *m_pipeline = NULL;
    GstElement *m_stream_muxer = NULL;

private:
    ConfigManager *m_config;
    user_callback_data *m_user_callback_data;
    void init_curl();
    void free_curl();
};
#endif