#include "FaceBin.h"
#include <nvdsinfer_custom_impl.h>
#include <nvds_obj_encode.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <librdkafka/rdkafkacpp.h>
#include <algorithm>
#include <cmath>
#include "message.h"

gpointer user_copy_facemark_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(data);
    NvDsFaceMetaData *facemark_meta_data_ptr = reinterpret_cast<NvDsFaceMetaData *>(
        user_meta->user_meta_data);
    NvDsFaceMetaData *new_facemark_meta_data_ptr = reinterpret_cast<NvDsFaceMetaData *>(
        g_memdup(facemark_meta_data_ptr, sizeof(NvDsFaceMetaData)));
    return reinterpret_cast<gpointer>(new_facemark_meta_data_ptr);
}

/**
 * @brief implement NvDsMetaReleaseFun
 *
 * @param data
 * @param user_data
 */
void user_release_facemark_data(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(data);
    NvDsFaceMetaData *facemark_meta_data_ptr = reinterpret_cast<NvDsFaceMetaData *>(
        user_meta->user_meta_data);
    delete facemark_meta_data_ptr;
}

static inline bool cmp(Detection &a, Detection &b)
{
    return a.class_confidence > b.class_confidence;
}

static inline float iou(float lbox[4], float rbox[4])
{
    float interBox[] = {
        std::max(lbox[0], rbox[0]), // left
        std::min(lbox[2], rbox[2]), // right
        std::max(lbox[1], rbox[1]), // top
        std::min(lbox[3], rbox[3]), // bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS /
           ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) +
            (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS + 0.000001f);
}

static void nms(std::vector<Detection> &res, float *output, float post_cluster_thresh = 0.7, float iou_threshold = 0.4)
{
    std::vector<Detection> dets;
    for (int i = 0; i < output[0]; i++)
    {
        if (output[15 * i + 1 + 4] <= post_cluster_thresh)
            continue;
        Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m)
    {
        auto &item = dets[m];
        res.push_back(item);
        // std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n)
        {
            if (iou(item.bbox, dets[n].bbox) > iou_threshold)
            {
                dets.erase(dets.begin() + n);
                --n;
            }
        }
    }
}

GstPadProbeReturn NvInferFaceBin::osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn NvInferFaceBin::pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buf, &inmap, GST_MAP_READ))
    {
        QDTLog::error("%s: %d input buffer mapinfo failed", __FILE__, __LINE__);
        return GST_PAD_PROBE_DROP;
    }
    NvBufSurface *ip_surf = (NvBufSurface *)inmap.data;
    gst_buffer_unmap(buf, &inmap);

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    // FIXME: should parse from config file
    NvDsInferParseDetectionParams detectionParams;
    detectionParams.numClassesConfigured = 1;
    detectionParams.perClassPreclusterThreshold = {0.1};
    detectionParams.perClassPostclusterThreshold = {0.7};

    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsMetaList *l_raw = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        // raw output when enable output-tensor-meta
        float muxer_output_height = frame_meta->pipeline_height;
        float muxer_output_width = frame_meta->pipeline_width;
        for (l_raw = frame_meta->frame_user_meta_list; l_raw != NULL; l_raw = l_raw->next)
        {
            NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_raw->data);
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
            {
                continue;
            }

            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta = reinterpret_cast<NvDsInferTensorMeta *>(user_meta->user_meta_data);
            float pgie_net_height = meta->network_info.height;
            float pgie_net_width = meta->network_info.width;

            VTX_ASSERT(meta->num_output_layers == 1); // we only have one output layer
            for (unsigned int i = 0; i < meta->num_output_layers; i++)
            {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
                if (meta->out_buf_ptrs_dev[i])
                {
                    cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                               info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                }
            }

            /* Parse output tensor, similar to NvDsInferParseCustomRetinaface  */
            std::vector<NvDsInferLayerInfo> outputLayersInfo(meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);
            NvDsInferLayerInfo outputLayerInfo = outputLayersInfo.at(0);
            // std::vector < NvDsInferObjectDetectionInfo > objectList;
            float *output = (float *)outputLayerInfo.buffer;

            std::vector<Detection> res;
            nms(res, output, detectionParams.perClassPostclusterThreshold[0]);

            // QDTLog::info("number of detection box = {}", res.size());
            /* Iterate final rectangules and attach result into frame's obj_meta_list */
            for (const auto &obj : res)
            {
                // if (obj.class_confidence > 0.63)
                {
                    NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);

                    // FIXME: a `meta` can produce more than once `obj`. Hence cannot set obj_meta->unique_component_id = meta->unique_id;
                    obj_meta->unique_component_id = meta->unique_id;
                    obj_meta->confidence = obj.class_confidence;

                    // untracked object. Set tracking_id to -1
                    obj_meta->object_id = UNTRACKED_OBJECT_ID;
                    obj_meta->class_id = FACE_CLASS_ID; // only have one class

                    /* retrieve bouding box */
                    float scale_x = muxer_output_width / pgie_net_width;
                    float scale_y = muxer_output_height / pgie_net_height;

                    obj_meta->detector_bbox_info.org_bbox_coords.left = obj.bbox[0];
                    obj_meta->detector_bbox_info.org_bbox_coords.top = obj.bbox[1];
                    obj_meta->detector_bbox_info.org_bbox_coords.width = obj.bbox[2] - obj.bbox[0];
                    obj_meta->detector_bbox_info.org_bbox_coords.height = obj.bbox[3] - obj.bbox[1];

                    obj_meta->detector_bbox_info.org_bbox_coords.left *= scale_x;
                    obj_meta->detector_bbox_info.org_bbox_coords.top *= scale_y;
                    obj_meta->detector_bbox_info.org_bbox_coords.width *= scale_x;
                    obj_meta->detector_bbox_info.org_bbox_coords.height *= scale_y;

                    // add padding to bbox found
                    int padding = 5;
                    float left, top, width, height;
                    obj_meta->detector_bbox_info.org_bbox_coords.left -= padding;
                    obj_meta->detector_bbox_info.org_bbox_coords.top -= padding;
                    obj_meta->detector_bbox_info.org_bbox_coords.width += 2 * padding;
                    obj_meta->detector_bbox_info.org_bbox_coords.height += 2 * padding;

                    left = obj_meta->detector_bbox_info.org_bbox_coords.left;
                    top = obj_meta->detector_bbox_info.org_bbox_coords.top;
                    width = obj_meta->detector_bbox_info.org_bbox_coords.width;
                    height = obj_meta->detector_bbox_info.org_bbox_coords.height;

                    left = left > 0 ? left : 0;
                    top = top > 0 ? top : 0;
                    width = (left + width < muxer_output_width) ? width : (muxer_output_width - left);
                    height = (top + height < muxer_output_height) ? height : (muxer_output_height - top);

                    obj_meta->detector_bbox_info.org_bbox_coords.left = left;
                    obj_meta->detector_bbox_info.org_bbox_coords.top = top;
                    obj_meta->detector_bbox_info.org_bbox_coords.width = width;
                    obj_meta->detector_bbox_info.org_bbox_coords.height = height;

                    /* for nvdosd */
                    NvOSD_RectParams &rect_params = obj_meta->rect_params;
                    NvOSD_TextParams &text_params = obj_meta->text_params;
                    rect_params.left = obj_meta->detector_bbox_info.org_bbox_coords.left;
                    rect_params.top = obj_meta->detector_bbox_info.org_bbox_coords.top;
                    rect_params.width = obj_meta->detector_bbox_info.org_bbox_coords.width;
                    rect_params.height = obj_meta->detector_bbox_info.org_bbox_coords.height;
                    rect_params.border_width = 3;
                    rect_params.has_bg_color = 0;
                    rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};

                    // store landmark in obj_user_meta_list
                    NvDsFaceMetaData *face_meta_ptr = new NvDsFaceMetaData();
                    for (int j = 0; j < 2 * NUM_FACEMARK; j += 2)
                    {
                        face_meta_ptr->stage = NvDsFaceMetaStage::DETECTED;
                        face_meta_ptr->faceMark[j] = obj.landmark[j];
                        face_meta_ptr->faceMark[j + 1] = obj.landmark[j + 1];
                        face_meta_ptr->faceMark[j] *= scale_x;
                        face_meta_ptr->faceMark[j + 1] *= scale_y;
                    }

                    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
                    user_meta->user_meta_data = static_cast<void *>(face_meta_ptr);
                    NvDsMetaType user_meta_type = (NvDsMetaType)NVDS_OBJ_USER_META_FACE;
                    user_meta->base_meta.meta_type = user_meta_type;
                    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)user_copy_facemark_meta;
                    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)user_release_facemark_data;
                    nvds_add_user_meta_to_obj(obj_meta, user_meta);
                    nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);

                    NvDsObjEncUsrArgs userData = {0};
                    userData.saveImg = FALSE;
                    userData.attachUsrMeta = TRUE;
                    /* Set if Image scaling Required */
                    userData.scaleImg = FALSE;
                    userData.scaledWidth = 0;
                    userData.scaledHeight = 0;
                    /* Quality */
                    userData.quality = 80;
                    /*Main Function Call */
                    nvds_obj_enc_process((NvDsObjEncCtxHandle)_udata, &userData, ip_surf, obj_meta, frame_meta);
                }
            }
        }
    }
    nvds_obj_enc_finish((NvDsObjEncCtxHandle)_udata);

    return GST_PAD_PROBE_OK;
}

static size_t WriteJsonCallback(char *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

void getFaceMetaData(NvDsFrameMeta *frame_meta, NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta,
                     user_callback_data *callback_data, NvDsInferLayerInfo *output_layer_info)
{
    NvDsFaceMsgData *face_msg_sub_meta = (NvDsFaceMsgData *)g_malloc0(sizeof(NvDsFaceMsgData));
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

            const int feature_size = output_layer_info->inferDims.numElements;
            float *cur_feature = reinterpret_cast<float *>(output_layer_info->buffer) +
                                 faceMeta->aligned_index * feature_size;
            memcpy(faceMeta->feature, cur_feature, feature_size * sizeof(float));
            face_msg_sub_meta->feature = g_strdup(b64encode(faceMeta->feature, FEATURE_SIZE));

            for (int i = 0; i < FEATURE_SIZE; i++)
            {
                callback_data->batch_face_feature.push_back(cur_feature[i]);
            }
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
        // Send HTTP request
        CURL *curl = callback_data->curl;
        std::string response_string;

        int num_face = callback_data->batch_face_feature.size() / FEATURE_SIZE;
        const char *data = gen_body(num_face, b64encode(callback_data->batch_face_feature.data(), FEATURE_SIZE * num_face));
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteJsonCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

        // request over HTTP/2, using the same connection!
        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK)
        {
            QDTLog::warn("curl from vector database failed");
        }
        else
        {
            // QDTLog::info("Response string = {}", response_string);

            std::vector<std::string> response_list = parseListJson(response_string);
            for (int i = 0; i < response_list.size(); i++)
            {
                Document doc;
                doc.Parse(response_list[i].c_str());
                Value &s = doc["score"];
                callback_data->face_meta_list[i]->confidence_score = s.GetDouble();
                s = doc["code"];
                callback_data->face_meta_list[i]->staff_id = g_strdup(s.GetString());
                s = doc["name"];
                callback_data->face_meta_list[i]->name = g_strdup(s.GetString());
            }
        }
        // Sending FaceRawMeta message to Kafka server
        for (int i = 0; i < callback_data->face_meta_list.size(); i++)
        {
            NvDsEventMsgMeta *meta_msg = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
            meta_msg->extMsg = (void *)callback_data->face_meta_list[i];
            meta_msg->extMsgSize = sizeof(NvDsEventMsgMeta);

            gchar *message = generate_FaceRawMeta_message(meta_msg);
            RdKafka::ErrorCode err = callback_data->kafka_producer->producer->produce(callback_data->face_rawmeta_topic,
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
                    if (callback_data->kafka_producer->counter > 10)
                    {
                        callback_data->kafka_producer->counter = 0;
                        callback_data->kafka_producer->producer->poll(100);
                    }
                }
            }
        }
        callback_data->frame_since_last_decode_face_name = 0;
        callback_data->batch_face_feature.clear();
        callback_data->face_meta_list.clear();
    }

    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        FILE *file;

        if (user_meta->base_meta.meta_type == NVDS_CROP_IMAGE_META)
        {
            if (face_msg_sub_meta->confidence_score && callback_data->face_feature_confidence_threshold &&
                std::string(face_msg_sub_meta->name) != std::string("Unknown") &&
                callback_data->save_crop_img)
            {
                NvDsObjEncOutParams *enc_jpeg_image =
                    (NvDsObjEncOutParams *)user_meta->user_meta_data;

                std::string fileNameString = "crop_img/" + std::to_string(frame_meta->frame_num) + "_" + std::to_string(face_msg_sub_meta->confidence_score) + std::string(face_msg_sub_meta->name) +
                                             "_" + std::to_string((int)obj_meta->rect_params.width) + "x" + std::to_string((int)obj_meta->rect_params.height) + ".jpg";

                /* Write to File */
                file = fopen(fileNameString.c_str(), "wb");
                fwrite(enc_jpeg_image->outBuffer, sizeof(uint8_t),
                       enc_jpeg_image->outLen, file);
                fclose(file);
            }
        }
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

void NvInferFaceBin::sgie_output_callback(GstBuffer *buf,
                                          NvDsInferNetworkInfo *network_info,
                                          NvDsInferLayerInfo *layers_info,
                                          guint num_layers,
                                          guint batch_size,
                                          gpointer user_data)
{
    user_callback_data *callback_data = reinterpret_cast<user_callback_data *>(user_data);
    callback_data->frame_since_last_decode_face_name++;
    /* Find the only output layer */
    NvDsInferLayerInfo *output_layer_info;
    NvDsInferLayerInfo *input_layer_info;
    for (int i = 0; i < num_layers; i++)
    {
        NvDsInferLayerInfo *info = &layers_info[i];
        if (info->isInput)
        {
            input_layer_info = info;
        }
        else
        {
            output_layer_info = info;
            // TODO: the info also include input tensor, which is the 3x112x112 input. COuld be use for something.
        }
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

    /* Assign feature to NvDsFaceMetaData */
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
                getFaceMetaData(frame_meta, batch_meta, obj_meta, callback_data, output_layer_info);
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

        // This is where to create the final NvDsEventMsgMeta before sending
        NvDsEventMsgMeta *meta_msg = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
        meta_msg->extMsg = (void *)msg_meta_content;
        meta_msg->extMsgSize = sizeof(XFaceMOTMsgMeta);
        meta_msg->componentId = 1;

        gchar *message = generate_MOTRawMeta_message(meta_msg);
        RdKafka::ErrorCode err = callback_data->kafka_producer->producer->produce(callback_data->mot_rawmeta_topic,
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

    NvBufSurfaceUnMap(surface, -1, -1);
    gst_buffer_unmap(buf, &in_map_info);
}

GstPadProbeReturn NvInferFaceBin::tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    return GST_PAD_PROBE_OK;
}
