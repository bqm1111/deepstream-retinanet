#include "videoSource.h"
#include "QDTLog.h"
#include <json-glib/json-glib.h>
#include <librdkafka/rdkafkacpp.h>

AppPipeline::~AppPipeline()
{
    // gst_object_unref(GST_OBJECT(m_pipeline));
    // gst_element_release_request_pad(m_tee, m_tee_msg_pad);
    // gst_element_release_request_pad(m_tee, m_tee_display_pad);
    // gst_object_unref(m_tee_msg_pad);
    // gst_object_unref(m_tee_display_pad);
}

void AppPipeline::create(std::string pipeline_name)
{
    m_pipeline_name = pipeline_name;
    m_pipeline = gst_pipeline_new(m_pipeline_name.c_str());
}

int AppPipeline::numVideoSrc()
{
    return m_video_source.size();
}

void AppPipeline::add_video_source(std::vector<std::vector<std::string>> video_info, std::vector<std::string> video_name)
{
    int cnt = 0;
    for (const auto &info : video_info)
    {
        std::string video_path = info[0];
        m_video_source[video_name[cnt]] = numVideoSrc() + 1;
        int source_id = numVideoSrc() - 1;
        if (info[2] == std::string("file"))
        {
            m_source.push_back(gst_element_factory_make("filesrc", ("file-source-" + std::to_string(source_id)).c_str()));
            if (fs::path(video_path).extension() == ".avi")
            {
                m_demux.push_back(gst_element_factory_make("tsdemux", ("tsdemux-" + std::to_string(source_id)).c_str()));
            }
            else if (fs::path(video_path).extension() == ".mp4")
            {
                m_demux.push_back(gst_element_factory_make("qtdemux", ("qtdemux-" + std::to_string(source_id)).c_str()));
            }
        }
        else if (info[2] == std::string("rtsp"))
        {
            m_source.push_back(gst_element_factory_make("rtspsrc", ("rtsp-source-" + std::to_string(source_id)).c_str()));
            g_object_set(m_source[source_id], "latency", 300, NULL);
            if (info[1] == "h265")
            {
                m_demux.push_back(gst_element_factory_make("rtph265depay", ("rtph265depay-" + std::to_string(source_id)).c_str()));
            }
            else if (info[1] == "h264")
            {
                m_demux.push_back(gst_element_factory_make("rtph264depay", ("rtph264depay-" + std::to_string(source_id)).c_str()));
            }
            else
            {
                QDTLog::error("Unknown encode type to create video parser\n");
            }
        }

        else
        {
            QDTLog::error("Unknown video input type\n");
        }

        GST_ASSERT(m_source[source_id]);
        GST_ASSERT(m_demux[source_id]);

        if (info[1] == "h265")
        {
            m_parser.push_back(gst_element_factory_make("h265parse", ("h265-parser-" + std::to_string(source_id)).c_str()));
        }
        else if (info[1] == "h264")
        {
            m_parser.push_back(gst_element_factory_make("h264parse", ("h264-parser-" + std::to_string(source_id)).c_str()));
        }
        else
        {
            QDTLog::error("Unknown encode type to create video parser\n");
        }
        GST_ASSERT(m_parser[source_id]);
        m_decoder.push_back(gst_element_factory_make("nvv4l2decoder", ("decoder-" + std::to_string(source_id)).c_str()));
        GST_ASSERT(m_decoder[source_id]);

        g_object_set(m_source[source_id], "location", video_path.c_str(), NULL);

        /* link */
        gst_bin_add_many(
            GST_BIN(m_pipeline), m_source[source_id], m_demux[source_id],
            m_parser[source_id], m_decoder[source_id], NULL);
        if (info[2] == std::string("file"))
        {
            if (!gst_element_link_many(m_source[source_id], m_demux[source_id], NULL))
            {
                gst_printerr("%s:%d could not link elements in camera source\n", __FILE__, __LINE__);
                throw std::runtime_error("");
            }
            // link tsdemux to h265parser
            g_signal_connect(m_demux[source_id], "pad-added", G_CALLBACK(addnewPad),
                             m_parser[source_id]);
        }
        else if (info[2] == std::string("rtsp"))
        {
            g_signal_connect(m_source[source_id], "pad-added", G_CALLBACK(addnewPad),
                             m_demux[source_id]);
            if (!gst_element_link_many(m_demux[source_id], m_parser[source_id], NULL))
            {
                gst_printerr("%s:%d could not link elements in camera source\n", __FILE__, __LINE__);
                throw std::runtime_error("");
            }
        }

        if (!gst_element_link_many(m_parser[source_id], m_decoder[source_id], NULL))
        {
            gst_printerr("%s:%d could not link elements in camera source\n", __FILE__, __LINE__);
            throw std::runtime_error("");
        }
        cnt++;
    }
}

void AppPipeline::setLiveSource(bool is_live)
{
    m_live_source = is_live;
}

static GstPadProbeReturn streammux_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    user_callback_data *callback_data = reinterpret_cast<user_callback_data *>(_udata);
    const auto p1 = std::chrono::system_clock::now();
    double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch()).count();
    callback_data->timestamp = timestamp;

    return GST_PAD_PROBE_OK;
}
void AppPipeline::linkMuxer(int muxer_output_width, int muxer_output_height)
{
    m_stream_muxer = gst_element_factory_make("nvstreammux", "streammuxer");
    g_object_set(m_stream_muxer, "width", muxer_output_width,
                 "height", muxer_output_height,
                 "batch-size", numVideoSrc(),
                 "buffer-pool-size", 40,
                 "nvbuf-memory-type", 3,
                 "batched-push-timeout", 220000,
                 "live-source", m_live_source,
                 NULL);
    GstPad *streammux_pad = gst_element_get_static_pad(m_stream_muxer, "src");
    GST_ASSERT(streammux_pad);
    gst_pad_add_probe(streammux_pad, GST_PAD_PROBE_TYPE_BUFFER, streammux_src_pad_buffer_probe,
                      m_user_callback_data, NULL);
    g_object_unref(streammux_pad);

    gst_bin_add(GST_BIN(m_pipeline), m_stream_muxer);

    for (int i = 0; i < numVideoSrc(); i++)
    {
        GstPad *decoder_srcpad = gst_element_get_static_pad(m_decoder[i], "src");
        GST_ASSERT(decoder_srcpad);

        GstPad *muxer_sinkpad = gst_element_get_request_pad(m_stream_muxer, ("sink_" + std::to_string(i)).c_str());
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

void AppPipeline::linkTwoBranch(GstElement *mot_bin, GstElement *face_bin)
{
    m_tee_app = gst_element_factory_make("tee", "nvsink-tee-app");
    m_queue_mot = gst_element_factory_make("queue", "nvtee-queue-mot");
    m_queue_face = gst_element_factory_make("queue", "nvtee-queue-face");
    gst_bin_add_many(GST_BIN(m_pipeline), m_tee_app, m_queue_mot, m_queue_face, NULL);

    if (!gst_element_link_many(m_stream_muxer, m_tee_app, NULL))
    {
        gst_printerr("%s:%d Could not link streammuxer with tee_app\n", __FILE__, __LINE__);
    }

    if (!gst_element_link_many(m_queue_mot, mot_bin, NULL))
    {
        gst_printerr("%s:%d Could not link queue_mot with mot_inferbin\n", __FILE__, __LINE__);
    }
    if (!gst_element_link_many(m_queue_face, face_bin, NULL))
    {
        gst_printerr("%s:%d Could not link queue_face with face_inferbin\n", __FILE__, __LINE__);
    }

    GstPad *sink_pad = gst_element_get_static_pad(m_queue_mot, "sink");
    GstPad *tee_app_mot_pad = gst_element_get_request_pad(m_tee_app, "src_%u");
    if (!tee_app_mot_pad)
    {
        g_printerr("%s:%d Unable to get request pads\n", __FILE__, __LINE__);
    }

    if (gst_pad_link(tee_app_mot_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }
    gst_object_unref(sink_pad);

    sink_pad = gst_element_get_static_pad(m_queue_face, "sink");
    GstPad *tee_app_face_pad = gst_element_get_request_pad(m_tee_app, "src_%u");
    if (!tee_app_mot_pad)
    {
        g_printerr("%s:%d Unable to get request pads\n", __FILE__, __LINE__);
    }

    if (gst_pad_link(tee_app_face_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }
    gst_object_unref(sink_pad);
}