#include "NvInferBinBase.h"

GstElement *NvInferBinBase::createInferPipeline(GstElement *pipeline)
{
    m_pipeline = pipeline;
    createVideoSinkBin();
    linkMsgBroker();
    createInferBin();
    GstElement *inferbin;
    getMasterBin(inferbin);
    gst_bin_add(GST_BIN(m_pipeline), inferbin);
    if (!gst_element_link_many(inferbin, m_tiler, NULL))
    {
        g_printerr("%s:%d Cant link infer bin to tiler\n", __FILE__, __LINE__);
    }
    return inferbin;
}

void NvInferBinBase::createVideoSinkBin()
{
    m_tiler = gst_element_factory_make("nvmultistreamtiler", std::string("sink-nvmultistreamtiler" + m_module_name).c_str());
    GST_ASSERT(m_tiler);
    g_object_set(G_OBJECT(m_tiler), "rows", m_params.tiler_rows, NULL);
    g_object_set(G_OBJECT(m_tiler), "columns", m_params.tiler_cols, NULL);
    g_object_set(G_OBJECT(m_tiler), "width", m_params.tiler_width, NULL);
    g_object_set(G_OBJECT(m_tiler), "height", m_params.tiler_height, NULL);
    m_convert = gst_element_factory_make("nvvideoconvert", std::string("video-convert" + m_module_name).c_str());
    GST_ASSERT(m_convert);

    m_osd = gst_element_factory_make("nvdsosd", std::string("sink-nvdsosd" + m_module_name).c_str());
    GST_ASSERT(m_osd);
    m_tee = gst_element_factory_make("tee", std::string("nvsink-tee" + m_module_name).c_str());
    m_queue_display = gst_element_factory_make("queue", std::string("nvtee-queue-display" + m_module_name).c_str());

    m_sink = gst_element_factory_make("nveglglessink", std::string("nv-sink" + m_module_name).c_str());
    GST_ASSERT(m_sink);
    g_object_set(G_OBJECT(m_sink), "sync", TRUE, NULL);
    gst_bin_add_many(GST_BIN(m_pipeline), m_tiler, m_convert, m_osd, m_tee, m_queue_display, m_sink, NULL);

    if (!gst_element_link_many(m_tiler, m_convert, m_osd, m_tee, NULL))
    {
        gst_printerr("Could not link tiler, osd and sink\n");
    }

    if (!gst_element_link_many(m_queue_display, m_sink, NULL))
    {
        gst_printerr("Could not link tiler, osd and sink\n");
    }

    GstPad *sink_pad = gst_element_get_static_pad(m_queue_display, "sink");
    m_tee_display_pad = gst_element_get_request_pad(m_tee, "src_%u");
    if (!m_tee_display_pad)
    {
        g_printerr("%s:%d Unable to get request pads\n", __FILE__, __LINE__);
    }

    if (gst_pad_link(m_tee_display_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }

    gst_object_unref(sink_pad);
}

void NvInferBinBase::createFileSinkBin(std::string location)
{
    m_tiler = gst_element_factory_make("nvmultistreamtiler", std::string("sink-nvmultistreamtiler" + m_module_name).c_str());
    GST_ASSERT(m_tiler);
    g_object_set(G_OBJECT(m_tiler), "rows", m_params.tiler_rows, NULL);
    g_object_set(G_OBJECT(m_tiler), "columns", m_params.tiler_cols, NULL);
    g_object_set(G_OBJECT(m_tiler), "width", m_params.tiler_width, NULL);
    g_object_set(G_OBJECT(m_tiler), "height", m_params.tiler_height, NULL);
    m_convert = gst_element_factory_make("nvvideoconvert", std::string("video-convert" + m_module_name).c_str());
    GST_ASSERT(m_convert);

    m_osd = gst_element_factory_make("nvdsosd", std::string("sink-nvdsosd" + m_module_name).c_str());
    GST_ASSERT(m_osd);
    m_tee = gst_element_factory_make("tee", std::string("nvsink-tee" + m_module_name).c_str());
    m_queue_display = gst_element_factory_make("queue", std::string("nvtee-queue-display" + m_module_name).c_str());
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
    gst_bin_add_many(GST_BIN(m_pipeline), m_tiler, m_convert, m_osd, m_tee, m_queue_display, m_file_convert,
                     m_capsfilter, m_nvv4l2h265enc, m_h265parse, m_file_muxer, m_sink, NULL);

    if (!gst_element_link_many(m_tiler, m_convert, m_osd, m_tee, NULL))
    {
        gst_printerr("%s:%d Could not link tiler, osd and sink\n", __FILE__, __LINE__);
    }

    if (!gst_element_link_many(m_queue_display, m_file_convert,
                               m_capsfilter, m_nvv4l2h265enc, m_h265parse, NULL))
    {
        gst_printerr("%s:%d Could not link elements\n", __FILE__, __LINE__);
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

    GstPad *sink_pad = gst_element_get_static_pad(m_queue_display, "sink");
    m_tee_display_pad = gst_element_get_request_pad(m_tee, "src_%u");
    if (!m_tee_display_pad)
    {
        g_printerr("%s:%d Unable to get request pads\n", __FILE__, __LINE__);
    }

    if (gst_pad_link(m_tee_display_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }

    gst_object_unref(sink_pad);
}

void NvInferBinBase::linkMsgBroker()
{
    m_msgconv = gst_element_factory_make("nvmsgconv", std::string("nvmsg-converter" + m_module_name).c_str());
    m_msgbroker = gst_element_factory_make("nvmsgbroker", std::string("nvmsg-broker" + m_module_name).c_str());
    m_queue_msg = gst_element_factory_make("queue", std::string("nvtee-queue-msg" + m_module_name).c_str());

    if (!m_msgbroker || !m_msgconv || !m_tee || !m_queue_display || !m_queue_msg)
    {
        g_printerr("%s:%dOne element could not be created. Exiting.\n", __FILE__, __LINE__);
    }

    gst_bin_add_many(GST_BIN(m_pipeline), m_queue_msg, m_msgconv, m_msgbroker, NULL);

    if (!gst_element_link_many(m_queue_msg, m_msgconv, m_msgbroker, NULL))
    {
        g_printerr("%s:%d Elements could not be linked \n", __FILE__, __LINE__);
    }

    GstPad *sink_pad = gst_element_get_static_pad(m_queue_msg, "sink");
    m_tee_msg_pad = gst_element_get_request_pad(m_tee, "src_%u");
    if (!m_tee_msg_pad)
    {
        g_printerr("Unable to get request pads\n");
    }

    if (gst_pad_link(m_tee_msg_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }
    gst_object_unref(sink_pad);
}