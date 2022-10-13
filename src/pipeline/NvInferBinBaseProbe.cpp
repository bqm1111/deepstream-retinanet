#include "NvInferBinBase.h"
#include <chrono>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include "message.h"
GstPadProbeReturn NvInferBinBase::osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
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
            // g_print(" %s:%d Tiler Average runtime: %f  Average FPS: %f \n", __FILE__, __LINE__, avg_runtime, avg_fps);
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
    cv::imwrite(filename, bgr_frame);
    // cv::resize(bgr_frame, bgr_frame, cv::Size(1280, 720));
    std::vector<int> encode_param;
    std::vector<uchar> encoded_buf;
    encode_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    encode_param.push_back(80);
    cv::imencode(".jpg", bgr_frame, encoded_buf, encode_param);

    XFaceVisualMsg *msg_meta_content = (XFaceVisualMsg *)g_malloc0(sizeof(XFaceVisualMsg));
    msg_meta_content->timestamp = callback_data->timestamp;
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
}

GstPadProbeReturn NvInferBinBase::tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
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
