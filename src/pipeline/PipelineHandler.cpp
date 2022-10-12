#include "PipelineHandler.h"
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
static gpointer XFace_msg_visual_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = (NvDsEventMsgMeta *)g_memdup(srcMeta, sizeof(NvDsEventMsgMeta));
    dstMeta->componentId = srcMeta->componentId;

    dstMeta->extMsg = g_malloc0(sizeof(XFaceVisualMsg));
    XFaceVisualMsg *srcExtMsg = (XFaceVisualMsg *)srcMeta->extMsg;
    XFaceVisualMsg *dstExtMsg = (XFaceVisualMsg *)dstMeta->extMsg;
    dstExtMsg->frameId = srcExtMsg->frameId;
    dstExtMsg->timestamp = srcExtMsg->timestamp;
    dstExtMsg->width = srcExtMsg->width;
    dstExtMsg->height = srcExtMsg->height;
    dstExtMsg->num_channel = srcExtMsg->num_channel;
    dstExtMsg->cameraId = g_strdup(srcExtMsg->cameraId);
    dstExtMsg->sessionId = g_strdup(srcExtMsg->sessionId);
    dstExtMsg->full_img = g_strdup(srcExtMsg->full_img);

    dstMeta->extMsgSize = srcMeta->extMsgSize;
    return dstMeta;
}

static void XFace_msg_visual_release_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

    if (srcMeta->extMsgSize > 0)
    {
        XFaceVisualMsg *srcExtMsg = (XFaceVisualMsg *)srcMeta->extMsg;
        g_free(srcExtMsg->cameraId);
        g_free(srcExtMsg->sessionId);
        g_free(srcExtMsg->full_img);

        srcMeta->extMsgSize = 0;
    }
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}
static gchar *generate_XFace_visual_message(NvDsEventMsgMeta *meta)
{
    JsonNode *rootNode;
    JsonObject *rootObj;
    JsonObject *propObj;
    JsonObject *jObj;

    gchar *message;
    rootObj = json_object_new();
    propObj = json_object_new();

    // add frame info
    XFaceVisualMsg *msg_meta_content = (XFaceVisualMsg *)meta->extMsg;
    // json_object_set_string_member(rootObj, "timestamp", msg_meta_content->timestamp);
    json_object_set_string_member(rootObj, "title", g_strdup("HDImage"));
    json_object_set_string_member(rootObj, "description", g_strdup("HDImage of each frame from video sources"));
    json_object_set_string_member(rootObj, "type", g_strdup("object"));

    // Required
    JsonArray *jVisualPropRequired = json_array_sized_new(8);
    json_array_add_string_element(jVisualPropRequired, g_strdup("timestamp"));
    json_array_add_string_element(jVisualPropRequired, g_strdup("camera_id"));
    json_array_add_string_element(jVisualPropRequired, g_strdup("frame_id"));
    json_array_add_string_element(jVisualPropRequired, g_strdup("session_id"));
    json_array_add_string_element(jVisualPropRequired, g_strdup("width"));
    json_array_add_string_element(jVisualPropRequired, g_strdup("height"));
    json_array_add_string_element(jVisualPropRequired, g_strdup("channel"));
    json_array_add_string_element(jVisualPropRequired, g_strdup("image"));

    json_object_set_array_member(propObj, "required", jVisualPropRequired);

    // timestamp
    jObj = json_object_new();
    json_object_set_string_member(jObj, "description", g_strdup("Time stamp of this event message"));
    json_object_set_string_member(jObj, "type", g_strdup("double"));
    json_object_set_double_member(jObj, "value", msg_meta_content->timestamp);

    json_object_set_object_member(propObj, "timestamp", jObj);

    // Camera_id
    jObj = json_object_new();
    json_object_set_string_member(jObj, "description", g_strdup("camera_id of this frame"));
    json_object_set_string_member(jObj, "type", g_strdup("string"));
    json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->cameraId));
    json_object_set_object_member(propObj, "camera_id", jObj);

    // Frame_id
    jObj = json_object_new();
    json_object_set_string_member(jObj, "description", g_strdup("frame_id of this frame"));
    json_object_set_string_member(jObj, "type", g_strdup("integer"));
    json_object_set_int_member(jObj, "value", msg_meta_content->frameId);
    json_object_set_object_member(propObj, "frame_id", jObj);

    // session_id
    jObj = json_object_new();
    json_object_set_string_member(jObj, "description", g_strdup("session_id of this frame"));
    json_object_set_string_member(jObj, "type", g_strdup("string"));
    json_object_set_string_member(jObj, "value", msg_meta_content->sessionId);
    json_object_set_object_member(propObj, "frame_id", jObj);
    // width
    jObj = json_object_new();
    json_object_set_string_member(jObj, "description", g_strdup("witdh of this frame"));
    json_object_set_string_member(jObj, "type", g_strdup("integer"));
    json_object_set_int_member(jObj, "value", msg_meta_content->width);
    json_object_set_object_member(propObj, "width", jObj);
    // height
    jObj = json_object_new();
    json_object_set_string_member(jObj, "description", g_strdup("height of this frame"));
    json_object_set_string_member(jObj, "type", g_strdup("integer"));
    json_object_set_int_member(jObj, "value", msg_meta_content->height);
    json_object_set_object_member(propObj, "height", jObj);
    // num_channel
    jObj = json_object_new();
    json_object_set_string_member(jObj, "description", g_strdup("number of channel of this frame"));
    json_object_set_string_member(jObj, "type", g_strdup("integer"));
    json_object_set_int_member(jObj, "value", msg_meta_content->num_channel);
    json_object_set_object_member(propObj, "channel", jObj);
    // bas264 encoded image
    jObj = json_object_new();
    json_object_set_string_member(jObj, "description", g_strdup("bas264 encoded image of this frame"));
    json_object_set_string_member(jObj, "type", g_strdup("bytes"));
    json_object_set_string_member(jObj, "value", msg_meta_content->full_img);
    json_object_set_object_member(propObj, "image", jObj);

    json_object_set_object_member(rootObj, "properties", propObj);
    // create root node
    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, rootObj);

    // create message
    message = json_to_string(rootNode, TRUE);

    json_node_free(rootNode);
    json_object_unref(rootObj);

    return message;
}

static GstPadProbeReturn jpegenc_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    KafkaProducer *producer = reinterpret_cast<KafkaProducer *>(_udata);
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GST_ASSERT(buf);
    if (!buf)
    {
        return GST_PAD_PROBE_OK;
    }
    GstMapInfo in_map_info;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    user_callback_data *callback_data = reinterpret_cast<user_callback_data *>(_udata);

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        const auto p1 = std::chrono::system_clock::now();
        cv::Mat frame = cv::Mat(frame_meta->source_frame_height, frame_meta->source_frame_width, CV_8UC3, in_map_info.data);
        cv::resize(frame, frame, cv::Size(1280, 720));
        std::vector<int> encode_param;
        std::vector<uchar> encoded_buf;
        encode_param.push_back(cv::IMWRITE_JPEG_QUALITY);
        encode_param.push_back(80);
        // cv::imencode(".jpg", frame, encoded_buf, encode_param);

        XFaceVisualMsg *msg_meta_content = (XFaceVisualMsg *)g_malloc0(sizeof(XFaceVisualMsg));
        msg_meta_content->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch()).count();
        msg_meta_content->cameraId = g_strdup(std::string(callback_data->video_name[frame_meta->source_id]).c_str());
        msg_meta_content->frameId = frame_meta->frame_num;
        msg_meta_content->sessionId = g_strdup(callback_data->session_id);
        msg_meta_content->full_img = g_strdup(b64encode((uint8_t *)in_map_info.data, in_map_info.size));
        msg_meta_content->width = 0;
        msg_meta_content->height = 0;
        msg_meta_content->num_channel = 3;

        NvDsEventMsgMeta *visual_event_msg = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
        visual_event_msg->extMsg = (void *)msg_meta_content;
        visual_event_msg->extMsgSize = sizeof(XFaceVisualMsg);
        visual_event_msg->componentId = 2;

        gchar *message = generate_XFace_visual_message(visual_event_msg);
        RdKafka::ErrorCode err = callback_data->kafka_producer->producer->produce(std::string("HDImage"),
                                                                                  RdKafka::Topic::PARTITION_UA,
                                                                                  RdKafka::Producer::RK_MSG_COPY,
                                                                                  (gchar *)message,
                                                                                  std::string(message).length(),
                                                                                  NULL, 0,
                                                                                  0, NULL, NULL);
        if (err != RdKafka::ERR_NO_ERROR)
        {
            std::cerr << "% Failed to produce to topic "
                      << ": "
                      << RdKafka::err2str(err) << std::endl;

            if (err == RdKafka::ERR__QUEUE_FULL)
            {
                /* If the internal queue is full, wait for
                 * messages to be delivered and then retry.
                 * The internal queue represents both
                 * messages to be sent and messages that have
                 * been sent or failed, awaiting their
                 * delivery report callback to be called.
                 *
                 * The internal queue is limited by the
                 * configuration property
                 * queue.buffering.max.messages */
                callback_data->kafka_producer->producer->poll(1000 /*block for max 1000ms*/);
            }
        }

        // // Pack EventMsgMeta into UserMeta
        // NvDsUserMeta *user_event_visual = nvds_acquire_user_meta_from_pool(batch_meta);
        // if (user_event_visual)
        // {
        //     user_event_visual->user_meta_data = (void *)visual_event_msg;
        //     user_event_visual->base_meta.meta_type = NVDS_EVENT_MSG_META;
        //     user_event_visual->base_meta.copy_func = (NvDsMetaCopyFunc)XFace_msg_visual_copy_func;
        //     user_event_visual->base_meta.release_func = (NvDsMetaReleaseFunc)XFace_msg_visual_release_func;

        //     nvds_add_user_meta_to_frame(frame_meta, user_event_visual);
        // }
    }

    return GST_PAD_PROBE_OK;
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
        //
        m_tee_split_src.push_back(gst_element_factory_make("tee", ("tee-split-visual" + std::to_string(source_id)).c_str()));
        GST_ASSERT(m_tee_split_src[source_id]);
        m_queue_tee_visual.push_back(gst_element_factory_make("queue", ("nvtee-queue-visual" + std::to_string(source_id)).c_str()));
        GST_ASSERT(m_queue_tee_visual[source_id]);
        m_queue_tee_infer.push_back(gst_element_factory_make("queue", ("nvtee-queue-infer" + std::to_string(source_id)).c_str()));
        GST_ASSERT(m_queue_tee_infer[source_id]);
        m_nvjpeg_encode.push_back(gst_element_factory_make("jpegenc", ("nvJpegenc" + std::to_string(source_id)).c_str()));
        GST_ASSERT(m_nvjpeg_encode[source_id]);

        m_msgconv.push_back(gst_element_factory_make("nvmsgconv", ("nvmsgconv" + std::to_string(source_id)).c_str()));
        GST_ASSERT(m_msgconv[source_id]);

        m_msgbroker.push_back(gst_element_factory_make("nvmsgbroker", ("nvmsgbroker" + std::to_string(source_id)).c_str()));
        GST_ASSERT(m_msgbroker[source_id]);

        gst_bin_add_many(
            GST_BIN(m_pipeline), m_tee_split_src[source_id], m_queue_tee_visual[source_id],
            m_queue_tee_infer[source_id], m_nvjpeg_encode[source_id], NULL);

        if (!gst_element_link_many(m_parser[source_id], m_decoder[source_id], m_tee_split_src[source_id], NULL))
        {
            gst_printerr("%s:%d could not link elements in camera source\n", __FILE__, __LINE__);
            throw std::runtime_error("");
        }
        // if (!gst_element_link_many(m_queue_tee_visual[source_id], NULL))
        // {
        //     gst_printerr("%s:%d Could not link elements\n", __FILE__, __LINE__);
        // }

        GstPad *jpegenc_src_pad = gst_element_get_static_pad(m_nvjpeg_encode[source_id], "src");
        GST_ASSERT(jpegenc_src_pad);
        gst_pad_add_probe(jpegenc_src_pad, GST_PAD_PROBE_TYPE_BUFFER, jpegenc_src_pad_buffer_probe, m_producer, NULL);
        gst_object_unref(jpegenc_src_pad);

        GstPad *sink_pad = gst_element_get_static_pad(m_queue_tee_visual[source_id], "sink");
        m_tee_visual_pad.push_back(gst_element_get_request_pad(m_tee_split_src[source_id], "src_%u"));
        if (!m_tee_visual_pad[source_id])
        {
            g_printerr("%s:%d Unable to get request pads\n", __FILE__, __LINE__);
        }

        if (gst_pad_link(m_tee_visual_pad[source_id], sink_pad) != GST_PAD_LINK_OK)
        {
            g_printerr("Unable to link tee and message converter\n");
            gst_object_unref(sink_pad);
        }

        gst_object_unref(sink_pad);

        sink_pad = gst_element_get_static_pad(m_queue_tee_infer[source_id], "sink");
        m_tee_infer_pad.push_back(gst_element_get_request_pad(m_tee_split_src[source_id], "src_%u"));
        if (!m_tee_infer_pad[source_id])
        {
            g_printerr("%s:%d Unable to get request pads\n", __FILE__, __LINE__);
        }

        if (gst_pad_link(m_tee_infer_pad[source_id], sink_pad) != GST_PAD_LINK_OK)
        {
            g_printerr("Unable to link tee and message converter\n");
            gst_object_unref(sink_pad);
        }

        gst_object_unref(sink_pad);

        g_object_set(G_OBJECT(m_msgconv[source_id]), "config", MSG_CONFIG_PATH, NULL);
        g_object_set(G_OBJECT(m_msgconv[source_id]), "msg2p-lib", KAFKA_MSG2P_LIB, NULL);
        g_object_set(G_OBJECT(m_msgconv[source_id]), "payload-type", NVDS_PAYLOAD_CUSTOM, NULL);
        g_object_set(G_OBJECT(m_msgconv[source_id]), "msg2p-newapi", 0, NULL);
        g_object_set(G_OBJECT(m_msgconv[source_id]), "frame-interval", 30, NULL);
        // g_object_set(G_OBJECT(m_visual_msgconv), "multiple-payloads", TRUE, NULL);

        g_object_set(G_OBJECT(m_msgconv[source_id]), "comp-id", 2, NULL);

        g_object_set(G_OBJECT(m_msgbroker[source_id]), "comp-id", 2, NULL);
        g_object_set(G_OBJECT(m_msgbroker[source_id]), "proto-lib", KAFKA_PROTO_LIB,
                     "conn-str", m_params.connection_str.c_str(), "sync", FALSE, NULL);
        g_object_set(G_OBJECT(m_msgbroker[source_id]), "topic", m_params.visual_topic.c_str(), NULL);

        cnt++;
    }
}

void AppPipeline::setLiveSource(bool is_live)
{
    m_live_source = is_live;
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

    // m_video_convert = gst_element_factory_make("nvvideoconvert", "video-converter");
    // m_capsfilter = gst_element_factory_make("capsfilter", std::string("sink-capsfilter-rgba").c_str());
    // GST_ASSERT(m_capsfilter);
    // GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=(string)RGBA");
    // GST_ASSERT(caps);
    // g_object_set(G_OBJECT(m_capsfilter), "caps", caps, NULL);

    // gst_bin_add_many(GST_BIN(m_pipeline), m_stream_muxer, m_video_convert, m_capsfilter, NULL);
    // if(!gst_element_link_many(m_stream_muxer, m_video_convert, m_capsfilter))
    // {
    //     QDTLog::error("{}:{}Cant link element", __FILE__, __LINE__);
    // }
    gst_bin_add(GST_BIN(m_pipeline), m_stream_muxer);

    for (int i = 0; i < numVideoSrc(); i++)
    {
        GstPad *decoder_srcpad = gst_element_get_static_pad(m_queue_tee_infer[i], "src");
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