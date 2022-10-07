#include "NvInferBinBase.h"
#include <chrono>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
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

static void sendFullFrameMsg(NvBufSurface *surface, NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, user_callback_data *callback_data)
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
    // cv::imwrite(filename, bgr_frame);
    cv::resize(bgr_frame, bgr_frame, cv::Size(100, 100));
    std::vector<int> encode_param;
    std::vector<uchar> encoded_buf;
    encode_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    encode_param.push_back(80);
    cv::imencode(".jpg", bgr_frame, encoded_buf, encode_param);
    // QDTLog::info("buffer size = {}", encoded_buf.size());
    const auto p1 = std::chrono::system_clock::now();

    // XFaceVisualMsg *msg_meta_content = (XFaceVisualMsg *)g_malloc0(sizeof(XFaceVisualMsg));
    // msg_meta_content->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch()).count();
    // msg_meta_content->cameraId = g_strdup(std::string(callback_data->video_name[frame_meta->source_id]).c_str());
    // msg_meta_content->frameId = frame_meta->frame_num;
    // msg_meta_content->sessionId = g_strdup(callback_data->session_id);
    // msg_meta_content->full_img = g_strdup(b64encode((uint8_t *)bgr_frame.data, bgr_frame.rows * bgr_frame.cols * 3));
    // msg_meta_content->width = bgr_frame.cols;
    // msg_meta_content->height = bgr_frame.rows;
    // msg_meta_content->num_channel = bgr_frame.channels();

    // NvDsEventMsgMeta *visual_event_msg = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
    // visual_event_msg->extMsg = (void *)msg_meta_content;
    // visual_event_msg->extMsgSize = sizeof(XFaceVisualMsg);
    // visual_event_msg->componentId = 2;

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

        sendFullFrameMsg(surface, batch_meta, frame_meta, callback_data);
    }
    NvBufSurfaceUnMap(surface, -1, -1);
    gst_buffer_unmap(buf, &in_map_info);

    return GST_PAD_PROBE_OK;
}
