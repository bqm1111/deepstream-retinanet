#include "NvInferBinBase.h"
#include <chrono>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include "message.h"
GstPadProbeReturn NvInferBinBase::timer_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
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
        sink_perf_struct->check_start();
        sink_perf_struct->update();
        sink_perf_struct->log();

        if (nvds_enable_latency_measurement)
        {
            NvDsFrameLatencyInfo *latency_info = (NvDsFrameLatencyInfo *)malloc(20 * sizeof(NvDsFrameLatencyInfo));
            int num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);
            for (int i = 0; i < num_sources_in_batch; i++)
            {
                QDTLOG_DEBUG("source_id={} frame_num={} frame latancy={}",
                             latency_info[i].source_id,
                             latency_info[i].frame_num,
                             latency_info[i].latency);
            }
            free(latency_info);
        }
    }

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn NvInferBinBase::tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    return GST_PAD_PROBE_OK;
}

static size_t WriteJsonCallback(char *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

void getFaceMetaData(NvDsFrameMeta *frame_meta, NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta,
                     user_callback_data *callback_data)
{
    std::shared_ptr<NvDsFaceMsgData> face_msg_sub_meta = std::make_shared<NvDsFaceMsgData>();
    face_msg_sub_meta->bbox.top = clip(obj_meta->rect_params.top / frame_meta->source_frame_height);
    face_msg_sub_meta->bbox.left = clip(obj_meta->rect_params.left / frame_meta->source_frame_width);
    face_msg_sub_meta->bbox.width = clip(obj_meta->rect_params.width / frame_meta->source_frame_width);
    face_msg_sub_meta->bbox.height = clip(obj_meta->rect_params.height / frame_meta->source_frame_height);

    // Generate timestamp
    face_msg_sub_meta->timestamp = g_strdup(callback_data->timestamp);
    face_msg_sub_meta->cameraId = g_strdup(std::string(callback_data->video_name[frame_meta->source_id]).c_str());
    face_msg_sub_meta->frameId = frame_meta->frame_num;
    face_msg_sub_meta->sessionId = g_strdup(callback_data->session_id);

    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
        {
            NvDsFaceMetaData *faceMeta = reinterpret_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);
            face_msg_sub_meta->feature = g_strdup(b64encode(faceMeta->feature, FEATURE_SIZE));
        }
        else if (user_meta->base_meta.meta_type == NVDS_CROP_IMAGE_META)
        {
            NvDsObjEncOutParams *enc_jpeg_image =
                (NvDsObjEncOutParams *)user_meta->user_meta_data;
            face_msg_sub_meta->encoded_img = g_strdup(b64encode(enc_jpeg_image->outBuffer, enc_jpeg_image->outLen));
        }
    }
    callback_data->face_meta_list.push_back(face_msg_sub_meta);
    
    // Wait until a certain amount of faces are received. Batching all of them to call a curl request to get their name
    if (callback_data->frame_since_last_decode_face_name > 5 || callback_data->face_meta_list.size() == 32)
    {
        // Sending FaceRawMeta message to Kafka server
        for (int i = 0; i < callback_data->face_meta_list.size(); i++)
        {
            gchar *message = generate_FaceRawMeta_message(callback_data->face_meta_list[i]);
            RdKafka::ErrorCode err = callback_data->meta_producer->producer->produce(callback_data->face_rawmeta_topic,
                                                                                     RdKafka::Topic::PARTITION_UA,
                                                                                     RdKafka::Producer::RK_MSG_FREE,
                                                                                     (gchar *)message,
                                                                                     std::string(message).length(),
                                                                                     NULL, 0,
                                                                                     0, NULL, NULL);
            callback_data->meta_producer->counter++;
             
            if (err != RdKafka::ERR_NO_ERROR)
            {
                if (err == RdKafka::ERR__QUEUE_FULL)
                {
                    if (callback_data->meta_producer->counter > 10)
                    {
                        callback_data->meta_producer->counter = 0;
                        callback_data->meta_producer->producer->poll(100);
                    }
                }
            }
        }
        callback_data->frame_since_last_decode_face_name = 0;
        callback_data->batch_face_feature.clear();
        callback_data->face_meta_list.clear();
    }
}

void getMOTMetaData(NvDsFrameMeta *frame_meta, NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta, std::vector<NvDsMOTMsgData *> &mot_meta_list)
{
    NvDsMOTMsgData *mot_msg_sub_meta = (NvDsMOTMsgData *)g_malloc0(sizeof(NvDsMOTMsgData));
    mot_msg_sub_meta->bbox.top = clip(obj_meta->rect_params.top / frame_meta->source_frame_height);
    mot_msg_sub_meta->bbox.left = clip(obj_meta->rect_params.left / frame_meta->source_frame_width);
    mot_msg_sub_meta->bbox.width = clip(obj_meta->rect_params.width / frame_meta->source_frame_width);
    mot_msg_sub_meta->bbox.height = clip(obj_meta->rect_params.height / frame_meta->source_frame_height);

    mot_msg_sub_meta->track_id = obj_meta->object_id;

    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_OBJ_USER_META_MOT)
        {
            NvDsMOTMetaData *motMeta = reinterpret_cast<NvDsMOTMetaData *>(user_meta->user_meta_data);
            mot_msg_sub_meta->embedding = g_strdup(motMeta->feature);
        }
    }
    mot_meta_list.push_back(mot_msg_sub_meta);
}

GstFlowReturn NvInferBinBase::newSampleCallback(GstElement *sink, gpointer *user_data)
{
    user_callback_data *callback_data = reinterpret_cast<user_callback_data *>(user_data);
    callback_data->frame_since_last_decode_face_name++;

    GstSample *sample;
    GstBuffer *buf = NULL;

    sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (gst_app_sink_is_eos(GST_APP_SINK(sink)))
    {
        QDTLog::info("EOS received in Appsink********\n");
    }

    if (sample)
    {
        /* Obtain GstBuffer from sample and then extract metadata from it. */
        buf = gst_sample_get_buffer(sample);
        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

        for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
            // QDTLog::info("width and height = {} - {}", frame_meta->source_frame_width, frame_meta->source_frame_height);
            std::vector<NvDsMOTMsgData *> mot_sub_meta_list;

            for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
            {
                NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
                if (obj_meta->class_id == FACE_CLASS_ID)
                {
                    getFaceMetaData(frame_meta, batch_meta, obj_meta, callback_data);
                }
                else if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
                {
                    getMOTMetaData(frame_meta, batch_meta, obj_meta, mot_sub_meta_list);
                }
            }

            // ===================================== XFace MetaData sent to Kafka =====================================
            XFaceMOTMsgMeta *msg_meta_content = (XFaceMOTMsgMeta *)g_malloc0(sizeof(XFaceMOTMsgMeta));
            // Get MOT meta
            msg_meta_content->num_mot_obj = mot_sub_meta_list.size();
            msg_meta_content->mot_meta_list = (NvDsMOTMsgData **)g_malloc0(mot_sub_meta_list.size() * sizeof(NvDsMOTMsgData *));
            memcpy(msg_meta_content->mot_meta_list, mot_sub_meta_list.data(), mot_sub_meta_list.size() * sizeof(NvDsMOTMsgData *));

            // Generate timestamp
            msg_meta_content->timestamp = g_strdup(callback_data->timestamp);
            msg_meta_content->cameraId = g_strdup(std::string(callback_data->video_name[frame_meta->source_id]).c_str());
            msg_meta_content->frameId = frame_meta->frame_num;
            msg_meta_content->sessionId = g_strdup(callback_data->session_id);

            gchar *message = generate_MOTRawMeta_message(msg_meta_content);
            RdKafka::ErrorCode err = callback_data->meta_producer->producer->produce(callback_data->mot_rawmeta_topic,
                                                                                     RdKafka::Topic::PARTITION_UA,
                                                                                     RdKafka::Producer::RK_MSG_FREE,
                                                                                     (gchar *)message,
                                                                                     std::string(message).length(),
                                                                                     NULL, 0,
                                                                                     0, NULL, NULL);
            freeXFaceMOTMsgMeta(msg_meta_content);
            callback_data->meta_producer->counter++;

            if (callback_data->meta_producer->counter > POLLING_COUNTER)
            {
                callback_data->meta_producer->counter = 0;
                callback_data->meta_producer->producer->poll(100);
            }
        }

        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }
    return GST_FLOW_ERROR;
}