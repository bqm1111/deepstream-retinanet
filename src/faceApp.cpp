#include "faceApp.h"
#include "DeepStreamAppConfig.h"
#include "message.h"

FaceApp::FaceApp(std::string app_name)
{
    m_pipeline = gst_pipeline_new(app_name.c_str());
    m_config = new ConfigManager();
    m_user_callback_data = new user_callback_data();
    m_user_callback_data->session_id = (gchar *)malloc(SESSION_ID_LENGTH);
    uuid_t uuid;
    uuid_generate_random(uuid);
    uuid_unparse_lower(uuid, m_user_callback_data->session_id);
}

FaceApp::~FaceApp()
{
    delete m_config;
    free_curl();
    free(m_user_callback_data->session_id);
    free(m_user_callback_data->timestamp);
    delete m_user_callback_data->kafka_producer;
    delete m_user_callback_data;
    if (!m_user_callback_data->trackers)
    {
        free(m_user_callback_data->trackers);
    }
}

void FaceApp::loadConfig()
{
    m_config->setContext();
    std::shared_ptr<DSAppConfig> appConf = std::dynamic_pointer_cast<DSAppConfig>(m_config->getConfig(ConfigType::DeepStreamApp));

    m_user_callback_data->muxer_output_height = appConf->getProperty(DSAppProperty::STREAMMUX_OUTPUT_WIDTH).toInt();
    m_user_callback_data->muxer_output_width = appConf->getProperty(DSAppProperty::STREAMMUX_OUTPUT_HEIGHT).toInt();
    m_user_callback_data->muxer_batch_size = appConf->getProperty(DSAppProperty::STREAMMUX_BATCH_SIZE).toInt();
    m_user_callback_data->muxer_buffer_pool_size = appConf->getProperty(DSAppProperty::STREAMMUX_BUFFER_POOL_SIZE).toInt();
    m_user_callback_data->muxer_nvbuf_memory_type = appConf->getProperty(DSAppProperty::STREAMMUX_NVBUF_MEMORY_TYPE).toInt();
    m_user_callback_data->muxer_live_source = appConf->getProperty(DSAppProperty::STREAMMUX_LIVE_SOURCE).toBool();
    m_user_callback_data->tiler_cols = appConf->getProperty(DSAppProperty::TILER_COLS).toInt();

    m_user_callback_data->tiler_rows = appConf->getProperty(DSAppProperty::TILER_ROWS).toInt();
    m_user_callback_data->tiler_width = appConf->getProperty(DSAppProperty::TILER_WIDTH).toInt();
    m_user_callback_data->tiler_height = appConf->getProperty(DSAppProperty::TILER_HEIGHT).toInt();
    m_user_callback_data->metadata_topic = appConf->getProperty(DSAppProperty::KAFKA_METADATA_TOPIC).toString();
    m_user_callback_data->visual_topic = appConf->getProperty(DSAppProperty::KAFKA_VISUAL_TOPIC).toString();
    m_user_callback_data->connection_str = appConf->getProperty(DSAppProperty::KAFKA_CONNECTION_STR).toString();
    m_user_callback_data->curl_address = appConf->getProperty(DSAppProperty::FACE_FEATURE_CURL_ADDRESS).toString();
    m_user_callback_data->face_feature_confidence_threshold = appConf->getProperty(DSAppProperty::FACE_CONFIDENCE_THRESHOLD).toFloat();
    m_user_callback_data->save_crop_img = appConf->getProperty(DSAppProperty::SAVE_CROP_IMG).toBool();

    init_curl();
    m_user_callback_data->kafka_producer = new KafkaProducer();
    m_user_callback_data->kafka_producer->init(m_user_callback_data->connection_str);
}

static GstPadProbeReturn streammux_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    user_callback_data *callback_data = reinterpret_cast<user_callback_data *>(_udata);
    const auto p1 = std::chrono::system_clock::now();
    double timestamp = std::chrono::duration_cast<std::chrono::microseconds>(p1.time_since_epoch()).count();
    callback_data->timestamp = (gchar *)malloc(MAX_TIME_STAMP_LEN);
    generate_ts_rfc3339(callback_data->timestamp, MAX_TIME_STAMP_LEN);
    return GST_PAD_PROBE_OK;
}

void FaceApp::addVideoSource(std::string list_video_src_file)
{
    parseJson(list_video_src_file, m_video_source_name, m_video_source_info);
    m_user_callback_data->video_name = m_video_source_name;

    std::vector<GstElement *> m_source;
    std::vector<GstElement *> m_demux;
    std::vector<GstElement *> m_parser;
    std::vector<GstElement *> m_decoder;
    int cnt = 0;
    for (const auto &info : m_video_source_info)
    {
        std::string video_path = info[0];
        int source_id = cnt++;
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
    }

    // Add streammuxer
    m_stream_muxer = gst_element_factory_make("nvstreammux", "streammuxer");
    GST_ASSERT(m_stream_muxer);
    g_object_set(m_stream_muxer, "width", m_user_callback_data->muxer_output_width,
                 "height", m_user_callback_data->muxer_output_height,
                 "batch-size", m_user_callback_data->muxer_batch_size,
                 "buffer-pool-size", m_user_callback_data->muxer_buffer_pool_size,
                 "nvbuf-memory-type", m_user_callback_data->muxer_nvbuf_memory_type,
                 "batched-push-timeout", 220000,
                 "live-source", TRUE,
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

    // Initialize trackers for MOT
    int num_tracker = numVideoSrc();
    m_user_callback_data->trackers = (tracker *)g_malloc0(sizeof(tracker) * num_tracker);
    for (size_t i = 0; i < num_tracker; i++)
        m_user_callback_data->trackers[i] = tracker(
            0.1363697015033318, 91, 0.7510890862625559, 18, 2, 1.);
}

void FaceApp::init_curl()
{
    m_user_callback_data->curl = curl_easy_init();
    CURL *m_curl = m_user_callback_data->curl;
    assert(m_curl);

    /* copy from postman */
    curl_easy_setopt(m_curl, CURLOPT_CUSTOMREQUEST, "POST");
    curl_easy_setopt(m_curl, CURLOPT_URL, m_user_callback_data->curl_address.c_str());

    // curl_easy_setopt(m_curl, CURLOPT_VERBOSE, 11);

    /* HTTP/2 */
    curl_easy_setopt(m_curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);

    /* No SSL */
    curl_easy_setopt(m_curl, CURLOPT_SSL_VERIFYPEER, 0);

    /* wait for pipe connection to confirm*/
    curl_easy_setopt(m_curl, CURLOPT_PIPEWAIT, 1L);

    curl_easy_setopt(m_curl, CURLOPT_TIMEOUT_MS, 200);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, headers);
}

void FaceApp::free_curl()
{
    curl_easy_cleanup(m_user_callback_data->curl);
}

GstElement *FaceApp::getPipeline()
{
    return m_pipeline;
}

int FaceApp::numVideoSrc()
{
    return m_video_source_name.size();
}

static void sendFullFrame(NvBufSurface *surface, NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, user_callback_data *callback_data)
{
    gint frame_width = (gint)surface->surfaceList[frame_meta->batch_id].width;
    gint frame_height = (gint)surface->surfaceList[frame_meta->batch_id].height;
    void *frame_data = surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0];
    size_t frame_step = surface->surfaceList[frame_meta->batch_id].pitch;

    cv::Mat frame = cv::Mat(frame_height, frame_width, CV_8UC4, frame_data, frame_step);
    cv::Mat bgr_frame;
    cv::cvtColor(frame, bgr_frame, cv::COLOR_RGBA2BGR);

    char filename[64];
    snprintf(filename, 64, "img/image%d_%d.jpg", frame_meta->source_id, frame_meta->frame_num);
    std::vector<int> encode_param;
    std::vector<uchar> encoded_buf;
    encode_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    encode_param.push_back(80);
    cv::imencode(".jpg", bgr_frame, encoded_buf, encode_param);

    XFaceVisualMsg *msg_meta_content = (XFaceVisualMsg *)g_malloc0(sizeof(XFaceVisualMsg));
    msg_meta_content->timestamp = g_strdup(callback_data->timestamp);
    msg_meta_content->cameraId = g_strdup(std::string(callback_data->video_name[frame_meta->source_id]).c_str());
    msg_meta_content->frameId = frame_meta->frame_num;
    msg_meta_content->sessionId = g_strdup(callback_data->session_id);
    msg_meta_content->full_img = g_strdup(b64encode((uchar *)encoded_buf.data(), encoded_buf.size()));

    msg_meta_content->width = bgr_frame.cols;
    msg_meta_content->height = bgr_frame.rows;
    msg_meta_content->num_channel = bgr_frame.channels();

    NvDsEventMsgMeta *visual_event_msg = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
    visual_event_msg->extMsg = (void *)msg_meta_content;
    visual_event_msg->extMsgSize = sizeof(XFaceVisualMsg);
    visual_event_msg->componentId = 2;

    gchar *message = generate_XFace_visual_message(visual_event_msg);
    RdKafka::ErrorCode err = callback_data->kafka_producer->producer->produce(callback_data->visual_topic,
                                                                              RdKafka::Topic::PARTITION_UA,
                                                                              RdKafka::Producer::RK_MSG_FREE,
                                                                              (gchar *)message,
                                                                              std::string(message).length(),
                                                                              NULL, 0,
                                                                              0, NULL, NULL);

    callback_data->kafka_producer->counter++;
    if (err != RdKafka::ERR_NO_ERROR)
    {
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
            if (callback_data->kafka_producer->counter > 10)
            {
                callback_data->kafka_producer->counter = 0;
                callback_data->kafka_producer->producer->poll(100);
            }
        }
    }
}

static GstPadProbeReturn capsfilter_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GST_ASSERT(buf);
    if (!buf)
    {
        return GST_PAD_PROBE_OK;
    }
    GstMapInfo in_map_info;
    NvBufSurface *surface = NULL;
    memset(&in_map_info, 0, sizeof(in_map_info));
    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ))
    {
        QDTLog::error("Error: Failed to map gst buffer\n");
        gst_buffer_unmap(buf, &in_map_info);
    }
    surface = (NvBufSurface *)in_map_info.data;
    NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE);
    NvBufSurfaceSyncForCpu(surface, -1, -1);

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    user_callback_data *callback_data = reinterpret_cast<user_callback_data *>(_udata);

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);

        sendFullFrame(surface, batch_meta, frame_meta, callback_data);
    }
    NvBufSurfaceUnMap(surface, -1, -1);
    gst_buffer_unmap(buf, &in_map_info);

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn fakesink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GST_ASSERT(buf);
    if (!buf)
    {
        return GST_PAD_PROBE_OK;
    }

    if (_udata != nullptr)
    {
        /* do speed mesurement */
        SinkPerfStruct *sink_perf_struct = reinterpret_cast<SinkPerfStruct *>(_udata);
        if (!sink_perf_struct->start_perf_measurement)
        {
            sink_perf_struct->start_perf_measurement = true;
            sink_perf_struct->last_tick = std::chrono::high_resolution_clock::now();
        }
        else
        {
            // Measure
            auto tick = std::chrono::high_resolution_clock::now();
            double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(tick - sink_perf_struct->last_tick).count();

            // Update
            sink_perf_struct->last_tick = tick;
            sink_perf_struct->total_time += elapsed_time;
            sink_perf_struct->num_ticks++;

            // Statistics
            double avg_runtime = sink_perf_struct->total_time / sink_perf_struct->num_ticks / 1e6;
            double avg_fps = 1.0 / avg_runtime;
            // QDTLog::info("Encode Average runtime: {}  Average FPS: {}", avg_runtime, avg_fps);
        }

        if (nvds_enable_latency_measurement)
        {
            NvDsFrameLatencyInfo latency_info[2];
            nvds_measure_buffer_latency(buf, latency_info);
            g_print(" %s Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
                    __func__,
                    latency_info[0].source_id,
                    latency_info[0].frame_num,
                    latency_info[0].latency);
        }
    }

    return GST_PAD_PROBE_OK;
}

void FaceApp::sequentialDetectAndMOT()
{
    // ======================== MOT BRANCH ========================
    std::shared_ptr<NvInferMOTBinConfig> mot_configs = std::make_shared<NvInferMOTBinConfig>(MOT_PGIE_CONFIG_PATH, MOT_SGIE_CONFIG_PATH);
    NvInferMOTBin mot_bin(mot_configs);
    // remember to acquire trackerList before createBin
    mot_bin.acquireUserData(m_user_callback_data);
    GstElement *mot_inferbin;
    mot_bin.createInferBin();
    mot_bin.getMasterBin(mot_inferbin);

    // ======================== DETECT BRANCH ========================
    std::shared_ptr<NvInferFaceBinConfig> face_configs = std::make_shared<NvInferFaceBinConfig>(FACEID_PGIE_CONFIG_PATH, FACEID_SGIE_CONFIG_PATH, FACEID_ALIGN_CONFIG_PATH);
    NvInferFaceBin face_bin(face_configs);
    // remember to acquire curl before createBin
    face_bin.acquireUserData(m_user_callback_data);
    GstElement *face_inferbin;
    face_bin.createInferBin();
    face_bin.getMasterBin(face_inferbin);

    // ========================================================================
    NvInferBinBase bin;
    bin.acquireUserData(m_user_callback_data);
    GstElement *tiler = bin.createNonInferPipeline(m_pipeline);

    GstElement *video_convert = gst_element_factory_make("nvvideoconvert", "video-converter");
    g_object_set(G_OBJECT(video_convert), "nvbuf-memory-type", 3, NULL);
    GstElement *capsfilter = gst_element_factory_make("capsfilter", std::string("sink-capsfilter-rgba").c_str());
    GST_ASSERT(capsfilter);
    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=(string)RGBA");
    GST_ASSERT(caps);
    g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);

    GstElement *tee = gst_element_factory_make("tee", "tee-split");
    GstElement *queue_infer = gst_element_factory_make("queue", "queue-infer");
    GstElement *queue_encode = gst_element_factory_make("queue", "queue-encode");

    GstElement *fakesink = gst_element_factory_make("fakesink", "osd");
    gst_bin_add_many(GST_BIN(m_pipeline), tee, queue_infer, queue_encode, video_convert, capsfilter, fakesink, NULL);
    gst_bin_add_many(GST_BIN(m_pipeline), face_inferbin, mot_inferbin, NULL);

    if (!gst_element_link_many(m_stream_muxer, tee, NULL))
    {
        QDTLog::error("Cannot link mot and face bin {}:{}", __FILE__, __LINE__);
    }

    if (!gst_element_link_many(queue_encode, video_convert, capsfilter, fakesink, NULL))
    {
        QDTLog::error("Cannot link mot and face bin {}:{}", __FILE__, __LINE__);
    }

    if (!gst_element_link_many(queue_infer, mot_inferbin, face_inferbin, bin.m_tiler, NULL))
    {
        QDTLog::error("Cannot link mot and face bin {}:{}", __FILE__, __LINE__);
    }

    // Link queue infer
    GstPad *sink_pad = gst_element_get_static_pad(queue_infer, "sink");
    GstPad *queue_infer_pad = gst_element_get_request_pad(tee, "src_%u");
    if (!queue_infer_pad)
    {
        g_printerr("Unable to get request pads\n");
    }

    if (gst_pad_link(queue_infer_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }
    gst_object_unref(sink_pad);

    // Link queue encode
    sink_pad = gst_element_get_static_pad(queue_encode, "sink");
    GstPad *queue_encode_pad = gst_element_get_request_pad(tee, "src_%u");
    if (!queue_encode_pad)
    {
        g_printerr("Unable to get request pads\n");
    }

    if (gst_pad_link(queue_encode_pad, sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Unable to link tee and message converter\n");
        gst_object_unref(sink_pad);
    }
    gst_object_unref(sink_pad);

    GstPad *capsfilter_src_pad = gst_element_get_static_pad(capsfilter, "src");
    GST_ASSERT(capsfilter_src_pad);
    gst_pad_add_probe(capsfilter_src_pad, GST_PAD_PROBE_TYPE_BUFFER, capsfilter_src_pad_buffer_probe,
                      m_user_callback_data, NULL);
    g_object_unref(capsfilter_src_pad);

    SinkPerfStruct *fakesink_perf = new SinkPerfStruct;
    GstPad *fakesink_pad = gst_element_get_static_pad(fakesink, "sink");
    GST_ASSERT(fakesink_pad);
    gst_pad_add_probe(fakesink_pad, GST_PAD_PROBE_TYPE_BUFFER, fakesink_pad_buffer_probe,
                      fakesink_perf, NULL);
    g_object_unref(fakesink_pad);

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}
