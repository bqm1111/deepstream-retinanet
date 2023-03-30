#ifndef APP_H
#define APP_H
#include <gst/app/gstappsink.h>
#include <string>
#include "FaceBin.h"
#include "MOTBin.h"
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
    FaceApp(int argc, char **argv);
    ~FaceApp();
    void init();
    void run();

    void loadConfig();
    void addVideoSource();
    void sequentialDetectAndMOT();
    GstElement *getPipeline();
    int numVideoSrc();
    std::string m_video_list;
    std::vector<std::string> m_video_source_name;
    std::vector<std::vector<std::string>> m_video_source_info;
    GstElement *m_pipeline = NULL;
    GstElement *m_stream_muxer = NULL;

    std::vector<GstElement *> m_source;
    std::vector<GstElement *> m_demux;
    std::vector<GstElement *> m_parser;
    std::vector<GstElement *> m_decoder;
    GstElement *m_osd;
    GstElement *m_tiler;
    GstElement *m_video_convert;
    GstElement *m_capsfilter;
    GstElement *m_tee;
    GstElement *m_queue_infer;
    GstElement *m_queue_encode;
    GstElement *m_fakesink;

    GstElement *m_mot_elem;
    GstElement *m_face_elem;
    NvInferMOTBin m_mot_bin;
    NvInferFaceBin m_face_bin;
    
    void freePipeline();
    GMainLoop *getMainloop() { return m_loop; }
    
private:
    GMainLoop *m_loop = NULL;
    GstBus *m_bus = NULL;
    guint m_bus_watch_id;
    ConfigManager *m_config;
    user_callback_data *m_user_callback_data;
    static gboolean bus_watch_callback(GstBus *_bus, GstMessage *_msg, gpointer _uData);
    void init_user_callback_data();
    void free_user_callback_data();
};
#endif