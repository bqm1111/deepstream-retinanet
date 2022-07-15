#include "app.h"
#include <gst/gstpad.h>

FaceApp::FaceApp()
{
    m_pipeline = gst_pipeline_new("app_pipeline");
}
FaceApp::~FaceApp()
{
    gst_object_unref(GST_OBJECT(m_pipeline));
}

int FaceApp::getNumVideoSource()
{
    return m_video_source_path.size();
}

void FaceApp::add_video_source(std::string video_path, std::string name)
{
    m_video_source_path.push_back(video_path);
    m_gst_vidsrc.push_back(GstVideoSrc(name));
}

void FaceApp::linkToShowVideo()
{
    for (int i = 0; i < m_gst_vidsrc.size(); i++)
    {
        m_gst_vidsrc[i].add_source_to_sink(m_video_source_path[i], i, m_pipeline);
    }
}

void FaceApp::linkMultipleSrcToMuxer()
{
    m_muxer = gst_element_factory_make("nvstreammux", "nvstreammux");
    g_object_set(m_muxer, "width", m_gstparam.muxer_output_width,
                 "height", m_gstparam.muxer_output_height,
                 "batch-size", m_video_source_path.size(),
                 "batched-push-timeout", 220000, // 5FPS
                 NULL);
    gst_bin_add(GST_BIN(m_pipeline), m_muxer);
    for (int i = 0; i < m_video_source_path.size(); i++)
    {
        GstPad *decoder_srcpad = gst_element_get_static_pad(m_gst_vidsrc[i].decoder, "src");
        GST_ASSERT(decoder_srcpad);

        GstPad *muxer_sinkpad = gst_element_get_request_pad(m_muxer, ("sink_" + std::to_string(i)).c_str());
        GST_ASSERT(muxer_sinkpad);

        GstPadLinkReturn pad_link_return = gst_pad_link(decoder_srcpad, muxer_sinkpad);
        if (GST_PAD_LINK_FAILED(pad_link_return))
        {
            gst_printerr("%s:%d could not link decoder and muxer, reason %d\n", __FILE__, __LINE__, pad_link_return);
            throw std::runtime_error("");
        }
        gst_object_unref(decoder_srcpad);
        gst_object_unref(muxer_sinkpad);
    }
}

void FaceApp::executePipeline()
{
}