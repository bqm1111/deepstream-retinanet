#include "VideoSource.h"
#include <c++/7/experimental/bits/fs_path.h>
#include <gst/gstcaps.h>
#include <gst/gstelementfactory.h>
#include <gst/gstpad.h>
#include <string>

static void VideoSource::newPadCB(GstElement *element, GstPad *pad, gpointer data)
{
    gchar *name;
    name = gst_pad_get_name(pad);

    GstCaps *p_caps = gst_pad_get_pad_template_caps(pad);
    GstElement *sink = GST_ELEMENT(data);
    if (gst_element_link_pads(element, name, sink, "sink") == false)
    {
        gst_print("newPadCB : failed to link elements\n");
        // throw std::runtime_error("");
    }
    g_free(name);
}

void VideoSource::add_source(std::string video_path, int source_id, GstElement *pipeline, GstElement *muxer)
{
    source = gst_element_factory_make("filesrc", ("file-source-" + std::to_string(source_id)).c_str());
    GST_ASSERT(source);

    if (fs::path(video_path).extension() == ".avi")
    {
        demux = gst_element_factory_make("tsdemux", ("tsdemux-" + std::to_string(source_id)).c_str());
    }
    else if (fs::path(video_path).extension() == ".mp4")
    {
        demux = gst_element_factory_make("qtdemux", ("qtdemux-" + std::to_string(source_id)).c_str());
    }
    parser = gst_element_factory_make("h265parse", ("h265-parser-" + std::to_string(source_id)).c_str());
    GST_ASSERT(parser);
    decoder = gst_element_factory_make("nvv4l2decoder", ("decoder-" + std::to_string(source_id)).c_str());
    GST_ASSERT(decoder);

    std::cout << "Input video path: " << video_path << std::endl;
    g_object_set(source, "location", video_path.c_str(), NULL);

    /* link */
    gst_bin_add_many(
        GST_BIN(pipeline), source, demux,
        parser, decoder, NULL);

    if (!gst_element_link_many(source, demux, NULL))
    {
        gst_printerr("%s:%d could not link elements in camera source\n", __FILE__, __LINE__);
        throw std::runtime_error("");
    }
    if (!gst_element_link_many(parser, decoder, NULL))
    {
        gst_printerr("%s:%d could not link elements in camera source\n", __FILE__, __LINE__);
        throw std::runtime_error("");
    }
    // link tsdemux to h265parser
    g_signal_connect(demux, "pad-added", G_CALLBACK(newPadCB),
                     parser);

    // link camera source to MOT bin
    GstPad *decoder_srcpad = gst_element_get_static_pad(decoder, "src");
    GST_ASSERT(decoder_srcpad);

    GstPad *muxer_sinkpad = gst_element_get_request_pad(muxer, ("sink_" + std::to_string(source_id)).c_str());
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