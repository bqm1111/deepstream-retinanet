#ifdef CUSTOM_SINK_HPP_6fa95f7498c9979378e83fed
#define CUSTOM_SINK_HPP_6fa95f7498c9979378e83fed
#include <string>
#include <gst/gst.h>
#include <chrono>
#include "common.h"

#ifndef MAX_DISPLAY_LEN
#define MAX_DISPLAY_LEN 64
#endif

static GstPadProbeReturn tiler_sink_pad_callback(GstPad *pad, GstPadProbeInfo *info, gpointer udata);
static GstPadProbeReturn osd_sink_pad_callback(GstPad *pad, GstPadProbeInfo *info, gpointer udata);

struct SinkConfig
{
    size_t muxer_batch_size;
    bool osd_pad = false;
    int tiler_rows;
    int tiler_cols;
    int tiler_width;
    int tiler_height;
};

int create_sink_bin(SinkConfig sinkConfig, std::string location, GstElement *pipeline, GstElement *last_element)
{
    GstElement *tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
    g_object_set(G_OBJECT(tiler), "rows", sinkConfig.tiler_rows, NULL);
    g_object_set(G_OBJECT(tiler), "columns", sinkConfig.tiler_cols, NULL);
    g_object_set(G_OBJECT(tiler), "width", sinkConfig.tiler_width, NULL);
    g_object_set(G_OBJECT(tiler), "height", sinkConfig.tiler_height, NULL);

    GstElement *convert1 = gst_element_factory_make("nvvideoconvert", "sink-converter");

    GST_ASSERT(convert1);
    GstElement *nvdosd = gst_element_factory_make("nvdosd", "sink-nvdosd");
    GST_ASSERT(nvdosd);

    GstElement *convert2 = gst_element_factory_make("nvvideoconvert", "sink-convert2");
    GST_ASSERT(convert2);

    GstElement * capsfilter = gst_element_factory_make("capsfilter", "sink-capsfilter");
    GST_ASSERT(capsfilter);

    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=(string)I420");
    GST_ASSERT(caps);

    g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
}
#endif