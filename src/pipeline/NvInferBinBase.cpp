#include "NvInferBinBase.h"
#include "QDTLog.h"

GstElement *NvInferBinBase::createInferPipeline(GstElement *pipeline)
{
    m_pipeline = pipeline;
    // createVideoSinkBin();
    createFileSinkBin("out.mkv");
    createInferBin();
    // linkMsgBroker();
    setMsgBrokerConfig();
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
    createFileSinkBin("out.mkv");
    linkMsgBroker();
    setMsgBrokerConfig();

    attachProbe();
    return m_tiler;
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
    m_metadata_msgconv = gst_element_factory_make("nvmsgconv", std::string("metadata-nvmsg-converter" + m_module_name).c_str());
    m_metadata_msgbroker = gst_element_factory_make("nvmsgbroker", std::string("metadata-nvmsg-broker" + m_module_name).c_str());
    m_queue_metadata_msg = gst_element_factory_make("queue", std::string("metadata-nvtee-queue-msg" + m_module_name).c_str());

    m_visual_msgconv = gst_element_factory_make("nvmsgconv", std::string("visual_nvmsg-converter" + m_module_name).c_str());
    m_visual_msgbroker = gst_element_factory_make("nvmsgbroker", std::string("visual_nvmsg-broker" + m_module_name).c_str());
    m_queue_visual_msg = gst_element_factory_make("queue", std::string("visual-nvtee-queue-msg" + m_module_name).c_str());

    if (!m_metadata_msgconv || !m_metadata_msgbroker || !m_queue_metadata_msg ||
        !m_visual_msgconv || !m_visual_msgbroker || !m_visual_msgbroker)
    {
        g_printerr("%s:%dOne element could not be created. Exiting.\n", __FILE__, __LINE__);
    }
    
    gst_bin_add_many(GST_BIN(m_pipeline), m_queue_metadata_msg, m_metadata_msgconv, m_metadata_msgbroker,
                     m_visual_msgconv, m_visual_msgbroker, m_queue_visual_msg, NULL);

    if (!gst_element_link_many(m_queue_metadata_msg, m_metadata_msgconv, m_metadata_msgbroker, NULL))
    {
        g_printerr("%s:%d Elements could not be linked \n", __FILE__, __LINE__);
    }

    if (!gst_element_link_many(m_queue_visual_msg, m_visual_msgconv, m_visual_msgbroker, NULL))
    {
        g_printerr("%s:%d Elements could not be linked \n", __FILE__, __LINE__);
    }
    // Link queue in metadata msgbroker branch
    GstPad *sink_pad = gst_element_get_static_pad(m_queue_metadata_msg, "sink");
    m_tee_metadata_msg_pad = gst_element_get_request_pad(m_tee, "src_%u");
    if (!m_tee_metadata_msg_pad)
    {
        g_printerr("Unable to get request pads\n");
    }

    if (gst_pad_link(m_tee_metadata_msg_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }
    gst_object_unref(sink_pad);

    // Link queue in visual msgbroker branch
    sink_pad = gst_element_get_static_pad(m_queue_visual_msg, "sink");
    m_tee_visual_msg_pad = gst_element_get_request_pad(m_tee, "src_%u");
    if (!m_tee_visual_msg_pad)
    {
        g_printerr("Unable to get request pads\n");
    }

    if (gst_pad_link(m_tee_visual_msg_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }
    gst_object_unref(sink_pad);
}

void NvInferBinBase::attachProbe()
{
    SinkPerfStruct *sink_perf = new SinkPerfStruct;
    GstPad *tiler_sink_pad = gst_element_get_static_pad(m_tiler, "sink");
    GST_ASSERT(tiler_sink_pad);
    gst_pad_add_probe(tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, tiler_sink_pad_buffer_probe,
                      sink_perf, NULL);
    g_object_unref(tiler_sink_pad);

    GstPad *osd_sink_pad = gst_element_get_static_pad(m_osd, "sink");
    GST_ASSERT(osd_sink_pad);
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe,
                      reinterpret_cast<gpointer>(m_tiler), NULL);
    gst_object_unref(osd_sink_pad);
}

void NvInferBinBase::setMsgBrokerConfig()
{
    // FACE and MOT Branch
    g_object_set(G_OBJECT(m_metadata_msgconv), "config", MSG_CONFIG_PATH, NULL);
    g_object_set(G_OBJECT(m_metadata_msgconv), "msg2p-lib", KAFKA_MSG2P_LIB, NULL);
    g_object_set(G_OBJECT(m_metadata_msgconv), "payload-type", NVDS_PAYLOAD_CUSTOM, NULL);
    g_object_set(G_OBJECT(m_metadata_msgconv), "msg2p-newapi", 0, NULL);
    g_object_set(G_OBJECT(m_metadata_msgconv), "frame-interval", 30, NULL);
    // g_object_set(G_OBJECT(m_metadata_msgconv), "multiple-payloads", TRUE, NULL);
    g_object_set(G_OBJECT(m_metadata_msgconv), "comp-id", 1, NULL);

    g_object_set(G_OBJECT(m_metadata_msgbroker), "comp-id", 1, NULL);
    g_object_set(G_OBJECT(m_metadata_msgbroker), "proto-lib", KAFKA_PROTO_LIB,
                 "conn-str", m_params.connection_str.c_str(), "sync", FALSE, NULL);
    g_object_set(G_OBJECT(m_metadata_msgbroker), "topic", m_params.metadata_topic.c_str(), NULL);

    // Crop image branch
    g_object_set(G_OBJECT(m_visual_msgconv), "config", MSG_CONFIG_PATH, NULL);
    g_object_set(G_OBJECT(m_visual_msgconv), "msg2p-lib", KAFKA_MSG2P_LIB, NULL);
    g_object_set(G_OBJECT(m_visual_msgconv), "payload-type", NVDS_PAYLOAD_CUSTOM, NULL);
    g_object_set(G_OBJECT(m_visual_msgconv), "msg2p-newapi", 0, NULL);
    g_object_set(G_OBJECT(m_visual_msgconv), "frame-interval", 30, NULL);
    // g_object_set(G_OBJECT(m_visual_msgconv), "multiple-payloads", TRUE, NULL);

    g_object_set(G_OBJECT(m_visual_msgconv), "comp-id", 2, NULL);

    g_object_set(G_OBJECT(m_visual_msgbroker), "comp-id", 2, NULL);
    g_object_set(G_OBJECT(m_visual_msgbroker), "proto-lib", KAFKA_PROTO_LIB,
                 "conn-str", m_params.connection_str.c_str(), "sync", FALSE, NULL);
    g_object_set(G_OBJECT(m_visual_msgbroker), "topic", m_params.visual_topic.c_str(), NULL);
}
