#include "PipelineHandler.h"
#include "common.h"
#include <c++/7/experimental/bits/fs_path.h>
#include <gst/gstcaps.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstinfo.h>
#include <gst/gstpad.h>
#include <gst/gstutils.h>
#include <string>

AppPipeline::AppPipeline(std::string pipeline_name, GstAppParam params)
{
    m_pipeline_name = pipeline_name;
    m_gstparams = params;
    m_pipeline = gst_pipeline_new(m_pipeline_name.c_str());
}

AppPipeline::~AppPipeline()
{
    gst_object_unref(GST_OBJECT(m_pipeline));
}

void AppPipeline::create(std::string pipeline_name, GstAppParam params)
{
    m_pipeline_name = pipeline_name;
    m_gstparams = params;
    m_pipeline = gst_pipeline_new(m_pipeline_name.c_str());
}

int AppPipeline::numVideoSrc()
{
    return m_video_source.size();
}

GstElement *AppPipeline::add_video_source(std::string video_path, std::string video_name)
{
    m_video_source[video_name] = numVideoSrc() + 1;
    int source_id = numVideoSrc() - 1;

    m_source.push_back(gst_element_factory_make("filesrc", ("file-source-" + std::to_string(source_id)).c_str()));
    GST_ASSERT(m_source[source_id]);

    if (fs::path(video_path).extension() == ".avi")
    {
        m_demux.push_back(gst_element_factory_make("tsdemux", ("tsdemux-" + std::to_string(source_id)).c_str()));
    }
    else if (fs::path(video_path).extension() == ".mp4")
    {
        m_demux.push_back(gst_element_factory_make("qtdemux", ("qtdemux-" + std::to_string(source_id)).c_str()));
    }
    
    m_parser.push_back(gst_element_factory_make("h265parse", ("h265-parser-" + std::to_string(source_id)).c_str()));
    GST_ASSERT(m_parser[source_id]);
    m_decoder.push_back(gst_element_factory_make("nvv4l2decoder", ("decoder-" + std::to_string(source_id)).c_str()));
    GST_ASSERT(m_decoder[source_id]);
    
    std::cout << "Input video path: " << video_path << std::endl;
    g_object_set(m_source[source_id], "location", video_path.c_str(), NULL);

    /* link */
    gst_bin_add_many(
        GST_BIN(m_pipeline), m_source[source_id], m_demux[source_id],
        m_parser[source_id], m_decoder[source_id], NULL);

    if (!gst_element_link_many(m_source[source_id], m_demux[source_id], NULL))
    {
        gst_printerr("%s:%d could not link elements in camera source\n", __FILE__, __LINE__);
        throw std::runtime_error("");
    }
    if (!gst_element_link_many(m_parser[source_id], m_decoder[source_id], NULL))
    {
        gst_printerr("%s:%d could not link elements in camera source\n", __FILE__, __LINE__);
        throw std::runtime_error("");
    }
    // link tsdemux to h265parser
    g_signal_connect(m_demux[source_id], "pad-added", G_CALLBACK(addnewPad),
                     m_parser[source_id]);

    return m_decoder[source_id];
}

void AppPipeline::linkMuxer()
{
    m_muxer = gst_element_factory_make("nvstreammux", "streammuxer");
    g_object_set(m_muxer, "width", m_gstparams.muxer_output_width,
                 "height", m_gstparams.muxer_output_height,
                 "batch-size", numVideoSrc(),
                 "batched-push-timeout", 220000, // 5FPS
                 NULL);
    gst_bin_add(GST_BIN(m_pipeline), m_muxer);

    for (int i = 0; i < numVideoSrc(); i++)
    {
        GstPad *decoder_srcpad = gst_element_get_static_pad(m_decoder[i], "src");
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

void AppPipeline::link(GstElement *in_elem, GstElement *out_elem)
{
    if (!gst_element_link_many(in_elem, out_elem, NULL))
    {
        gst_printerr("Could not link elements: %s%d\n", __FILE__, __LINE__);
    }
}

GstElement *AppPipeline::createGeneralSinkBin()
{
    m_tiler = gst_element_factory_make("nvmultistreamtiler", "sink-nvmultistreamtiler");
    GST_ASSERT(m_tiler);
    g_object_set(G_OBJECT(m_tiler), "rows", m_gstparams.tiler_rows, NULL);
    g_object_set(G_OBJECT(m_tiler), "columns", m_gstparams.tiler_cols, NULL);
    g_object_set(G_OBJECT(m_tiler), "width", m_gstparams.tiler_width, NULL);
    g_object_set(G_OBJECT(m_tiler), "height", m_gstparams.tiler_height, NULL);
    // m_osd = gst_element_factory_make("nvdsosd", "sink-nvdsosd");
    // GST_ASSERT(m_osd);

    m_sink = gst_element_factory_make("nveglglessink", "nv-sink");
    GST_ASSERT(m_sink);
    gst_bin_add_many(GST_BIN(m_pipeline), m_tiler, m_sink, NULL);

    if (!gst_element_link_many(m_tiler,  m_sink, NULL))
    {
        gst_printerr("Could not link tiler, osd and sink\n");
    }

    // {
    //     GstPad *osd_sink_pad = gst_element_get_static_pad(m_osd, "sink");
    //     GST_ASSERT(osd_sink_pad);
    //     gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_callback,
    //                       reinterpret_cast<gpointer>(m_tiler), NULL);
    //     gst_object_unref(osd_sink_pad);
    // }
    return m_sink;
}
