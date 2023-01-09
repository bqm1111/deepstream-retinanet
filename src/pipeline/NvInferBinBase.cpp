#include "NvInferBinBase.h"
#include "QDTLog.h"

GstElement *NvInferBinBase::createInferPipeline(GstElement *pipeline)
{
    m_pipeline = pipeline;
    // createVideoSinkBin();
    createFileSinkBin("out.mkv");
    createInferBin();
    GstElement *inferbin;
    getMasterBin(inferbin);
    gst_bin_add(GST_BIN(m_pipeline), inferbin);
    attachProbe();
    if (!gst_element_link_many(inferbin, m_tiler, NULL))
    {
        g_printerr("%s:%d Cant link infer bin to tiler\n", __FILE__, __LINE__);
    }
    return inferbin;
}

GstElement *NvInferBinBase::createNonInferPipeline(GstElement *pipeline)
{
    m_pipeline = pipeline;
    // createVideoSinkBin();
    // createFileSinkBin("out.mkv");
    createAppSinkBin();
    attachProbe();
    return m_osd;
}

void NvInferBinBase::createVideoSinkBin()
{
    m_tiler = gst_element_factory_make("nvmultistreamtiler", std::string("sink-nvmultistreamtiler" + m_module_name).c_str());
    GST_ASSERT(m_tiler);
    g_object_set(G_OBJECT(m_tiler), "rows", m_user_callback_data->tiler_rows, NULL);
    g_object_set(G_OBJECT(m_tiler), "columns", m_user_callback_data->tiler_cols, NULL);
    g_object_set(G_OBJECT(m_tiler), "width", m_user_callback_data->tiler_width, NULL);
    g_object_set(G_OBJECT(m_tiler), "height", m_user_callback_data->tiler_height, NULL);
    m_convert = gst_element_factory_make("nvvideoconvert", std::string("video-convert" + m_module_name).c_str());
    GST_ASSERT(m_convert);

    m_osd = gst_element_factory_make("nvdsosd", std::string("sink-nvdsosd" + m_module_name).c_str());
    GST_ASSERT(m_osd);

    m_sink = gst_element_factory_make("nveglglessink", std::string("nv-sink" + m_module_name).c_str());
    GST_ASSERT(m_sink);

    gst_bin_add_many(GST_BIN(m_pipeline), m_tiler, m_convert, m_osd, m_sink, NULL);

    if (!gst_element_link_many(m_tiler, m_convert, m_osd, m_sink, NULL))
    {
        gst_printerr("Could not link tiler, osd and sink\n");
    }
}

void NvInferBinBase::createAppSinkBin()
{
    m_queue = gst_element_factory_make("queue", std::string("sink-queue" + m_module_name).c_str());
    GST_ASSERT(m_queue);

    m_sink = gst_element_factory_make("appsink", std::string("appsink" + m_module_name).c_str());
    GST_ASSERT(m_sink);
    gst_app_sink_set_drop((GstAppSink *)m_sink, true);
    g_object_set(m_sink, "emit-signals", TRUE, "async", TRUE, "sync", FALSE, "blocksize", 4096000, NULL);

    /* Callback to access buffer and object info. */
    g_signal_connect(m_sink, "new-sample", G_CALLBACK(newSampleCallback), m_user_callback_data);

    gst_bin_add_many(GST_BIN(m_pipeline), m_queue, m_sink, NULL);

    if (!gst_element_link_many(m_queue, m_sink, NULL))
    {
        gst_printerr("Could not link tiler, osd and sink\n");
    }
}

void NvInferBinBase::createFileSinkBin(std::string location)
{
    m_tiler = gst_element_factory_make("nvmultistreamtiler", std::string("sink-nvmultistreamtiler" + m_module_name).c_str());
    GST_ASSERT(m_tiler);
    g_object_set(G_OBJECT(m_tiler), "rows", m_user_callback_data->tiler_rows, NULL);
    g_object_set(G_OBJECT(m_tiler), "columns", m_user_callback_data->tiler_cols, NULL);
    g_object_set(G_OBJECT(m_tiler), "width", m_user_callback_data->tiler_width, NULL);
    g_object_set(G_OBJECT(m_tiler), "height", m_user_callback_data->tiler_height, NULL);
    m_convert = gst_element_factory_make("nvvideoconvert", std::string("video-convert" + m_module_name).c_str());
    GST_ASSERT(m_convert);

    m_osd = gst_element_factory_make("nvdsosd", std::string("sink-nvdsosd" + m_module_name).c_str());
    GST_ASSERT(m_osd);
    GstElement *m_file_convert = gst_element_factory_make("nvvideoconvert", std::string("sink-nvvideoconvert2" + m_module_name).c_str());
    GST_ASSERT(m_file_convert);

    GstElement *m_capsfilter = gst_element_factory_make("capsfilter", std::string("sink-capsfilter" + m_module_name).c_str());
    GST_ASSERT(m_capsfilter);
    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=(string)I420");
    GST_ASSERT(caps);
    g_object_set(G_OBJECT(m_capsfilter), "caps", caps, NULL);

    GstElement *m_nvv4l2h265enc = gst_element_factory_make("nvv4l2h265enc", std::string("sink-nvv4l2h265enc" + m_module_name).c_str());
    GST_ASSERT(m_nvv4l2h265enc);

    GstElement *m_h265parse = gst_element_factory_make("h265parse", std::string("sink-h265parse" + m_module_name).c_str());
    GST_ASSERT(m_h265parse);

    GstElement *m_file_muxer = gst_element_factory_make("matroskamux", std::string("sink-muxer" + m_module_name).c_str());
    GST_ASSERT(m_file_muxer);

    GstElement *m_sink = gst_element_factory_make("filesink", std::string("sink-filesink" + m_module_name).c_str());
    GST_ASSERT(m_sink);
    g_object_set(G_OBJECT(m_sink), "location", location.c_str(), NULL);
    g_object_set(G_OBJECT(m_sink), "sync", false, NULL);
    g_object_set(G_OBJECT(m_sink), "async", true, NULL);

    g_object_set(G_OBJECT(m_sink), "sync", TRUE, NULL);
    gst_bin_add_many(GST_BIN(m_pipeline), m_tiler, m_convert, m_osd, m_file_convert,
                     m_capsfilter, m_nvv4l2h265enc, m_h265parse, m_file_muxer, m_sink, NULL);

    if (!gst_element_link_many(m_tiler, m_convert, m_osd, m_file_convert,
                               m_capsfilter, m_nvv4l2h265enc, m_h265parse, NULL))
    {
        gst_printerr("%s:%d Could not link tiler, osd and sink\n", __FILE__, __LINE__);
    }

    if (!gst_element_link_many(m_file_muxer, m_sink, NULL))
    {
        gst_printerr("%s:%dCould not link elements\n", __FILE__, __LINE__);
    }

    GstPad *muxer_sinkpad = gst_element_get_request_pad(m_file_muxer, "video_0");
    GST_ASSERT(muxer_sinkpad);

    GstPad *h265parse_srcpad = gst_element_get_static_pad(m_h265parse, "src");
    GstPadLinkReturn pad_link_return = gst_pad_link(h265parse_srcpad, muxer_sinkpad);

    if (GST_PAD_LINK_FAILED(pad_link_return))
    {
        gst_printerr("%s:%d could not link h265parse and matroskamux, reason %d\n", __FILE__, __LINE__, pad_link_return);
        throw std::runtime_error("");
    }
    gst_object_unref(muxer_sinkpad);
}

void NvInferBinBase::attachProbe()
{
    GstPad *sink_pad = gst_element_get_static_pad(m_queue, "sink");
    GST_ASSERT(sink_pad);
    gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER, timer_sink_pad_buffer_probe,
                      m_user_callback_data->fakesink_perf, NULL);

    gst_object_unref(sink_pad);
}
