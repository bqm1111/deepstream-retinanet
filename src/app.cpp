#include "app.h"
#include "face_bin.h"
#include <gst/gstpad.h>
#include <gst/gstpipeline.h>

FaceApp::FaceApp(std::string name)
{
    m_app_name = name;
    m_app.create(m_app_name, m_gstparam);
}

FaceApp::~FaceApp()
{
}

GstElement *FaceApp::getPipeline()
{
    return m_app.m_pipeline;
}
void FaceApp::add_video(std::string video_path, std::string video_name)
{
    m_app.add_video_source(video_path, video_name);
}

void FaceApp::linkMuxer()
{
    m_app.linkMuxer();
}
void FaceApp::showVideo()
{
    m_app.linkMuxer();
    m_app.createGeneralSinkBin();
    m_app.link(m_app.m_muxer, m_app.m_tiler);
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_app.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "showVideo");
}

void FaceApp::faceDetection()
{
    m_app.linkMuxer();
    m_app.createGeneralSinkBin();
    FaceBinConfigs face_configs{FACEID_PGIE_CONFIG_PATH, FACEID_ALIGN_CONFIG_PATH, FACEID_SGIE_CONFIG_PATH};
    FaceBin face_bin(face_configs);

    GstElement *bin = NULL;
    face_bin.getMasterBin(bin);
    gst_bin_add(GST_BIN(m_app.m_pipeline), bin);
    if (!gst_element_link_many(m_app.m_muxer, bin, m_app.m_tiler, NULL))
    {
        g_printerr("%s:%d Cant link face detection bin\n", __FILE__, __LINE__);
    }
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_app.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}

void FaceApp::detectAndSend()
{
    m_app.linkMuxer();
    m_app.createGeneralSinkBin();
    m_app.linkMsgBroker();
    FaceBinConfigs face_configs{FACEID_PGIE_CONFIG_PATH, FACEID_ALIGN_CONFIG_PATH, FACEID_SGIE_CONFIG_PATH};
    FaceBin face_bin(face_configs);

    GstElement *bin = NULL;
    face_bin.getMasterBin(bin);
    gst_bin_add(GST_BIN(m_app.m_pipeline), bin);
    if (!gst_element_link_many(m_app.m_muxer, bin, m_app.m_tiler, NULL))
    {
        g_printerr("%s:%d Cant link face detection bin\n", __FILE__, __LINE__);
    }
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_app.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}
