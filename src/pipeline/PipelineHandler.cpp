#include "PipelineHandler.h"
#include "QDTLog.h"

AppPipeline::AppPipeline(std::string pipeline_name)
{
    m_pipeline_name = pipeline_name;
    m_pipeline = gst_pipeline_new(m_pipeline_name.c_str());
}

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

void AppPipeline::add_video_source(std::map<std::string, std::string> video_info, std::vector<std::string> video_name)
{
    int cnt = 0;
    for (const auto &info : video_info)
    {
        std::string video_path = info.first;
        m_video_source[video_name[cnt]] = numVideoSrc() + 1;
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

        if (info.second == "h265")
        {
            m_parser.push_back(gst_element_factory_make("h265parse", ("h265-parser-" + std::to_string(source_id)).c_str()));
        }
        else if (info.second == "h264")
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

        QDTLog::info("Input video path = {}\n", video_path);
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

        cnt++;
    }
}

void AppPipeline::linkMuxer(int muxer_output_width, int muxer_output_height)
{
    m_stream_muxer = gst_element_factory_make("nvstreammux", "streammuxer");
    g_object_set(m_stream_muxer, "width", muxer_output_width,
                 "height", muxer_output_height,
                 "batch-size", numVideoSrc(),
                 "batched-push-timeout", 220000, // 5FPS
                 NULL);
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
