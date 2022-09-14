#include "faceApp.h"
#include "mot_bin.h"

FaceApp::FaceApp(std::string name)
{
    m_pipeline_name = name;
    m_pipeline.create(m_pipeline_name, m_gstparam);
}

FaceApp::~FaceApp()
{
}

GstElement *FaceApp::getPipeline()
{
    return m_pipeline.m_pipeline;
}
void FaceApp::add_video(std::string video_path, std::string video_name)
{
    m_video_source_path.push_back(video_path);
    m_pipeline.add_video_source(video_path, video_name);
}

void FaceApp::linkMuxer()
{
    m_pipeline.linkMuxer();
}

void FaceApp::showVideo()
{
    m_pipeline.linkMuxer();
    m_pipeline.createVideoSinkBin();
    m_pipeline.link(m_pipeline.m_stream_muxer, m_pipeline.m_tiler);
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pipeline.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "showVideo");
}

void FaceApp::faceDetection()
{
    m_pipeline.linkMuxer();
    m_pipeline.createVideoSinkBin();
    // m_app.createFileSinkBin("out.mp4");
    FaceBinConfigs face_configs{FACEID_PGIE_CONFIG_PATH, FACEID_ALIGN_CONFIG_PATH, FACEID_SGIE_CONFIG_PATH};
    FaceBin face_bin(face_configs);

    GstElement *bin = NULL;
    face_bin.getMasterBin(bin);
    gst_bin_add(GST_BIN(m_pipeline.m_pipeline), bin);
    if (!gst_element_link_many(m_pipeline.m_stream_muxer, bin, m_pipeline.m_tiler, NULL))
    {
        g_printerr("%s:%d Cant link face detection bin\n", __FILE__, __LINE__);
    }
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pipeline.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}

void FaceApp::detectAndSend()
{
    m_pipeline.linkMuxer();
    m_pipeline.createVideoSinkBin();
    m_pipeline.linkMsgBroker();
    FaceBinConfigs face_configs{FACEID_PGIE_CONFIG_PATH, FACEID_ALIGN_CONFIG_PATH, FACEID_SGIE_CONFIG_PATH};
    FaceBin face_bin(face_configs);

    GstElement *bin = NULL;
    face_bin.getMasterBin(bin);
    gst_bin_add(GST_BIN(m_pipeline.m_pipeline), bin);
    if (!gst_element_link_many(m_pipeline.m_stream_muxer, bin, m_pipeline.m_tiler, NULL))
    {
        g_printerr("%s:%d Cant link face detection bin\n", __FILE__, __LINE__);
    }
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pipeline.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}

void FaceApp::MOT()
{
    m_pipeline.linkMuxer();
    m_pipeline.createVideoSinkBin();
    // m_app.createFileSinkBin("out.mp4");
    MOTBinConfigs mot_configs{m_video_source_path.size(), MOT_PGIE_CONFIG_PATH, MOT_SGIE_CONFIG_PATH};
    MOTBin mot_bin(mot_configs);

    GstElement *bin = NULL;
    mot_bin.getMasterBin(bin);
    gst_bin_add(GST_BIN(m_pipeline.m_pipeline), bin);
    if (!gst_element_link_many(m_pipeline.m_stream_muxer, bin, m_pipeline.m_tiler, NULL))
    {
        g_printerr("%s:%d Cant link mot bin\n", __FILE__, __LINE__);
    }
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pipeline.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}