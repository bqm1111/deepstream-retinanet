#include "app.h"
#include <gst/gstpad.h>

FaceApp::FaceApp(std::string name)
{
    m_app_name = name;
    m_app.create(m_app_name, m_gstparam);
}

FaceApp::~FaceApp()
{
}


void FaceApp::add_video(std::string video_path, std::string video_name)
{
    m_app.add_video_source(video_path, video_name);
}

void FaceApp::showVideo()
{
    m_app.linkMuxer();
    m_app.createGeneralSinkBin();
    m_app.link(m_app.m_muxer, m_app.m_tiler);
}
