#include "FaceBin.h"
#include <nvdsinfer_custom_impl.h>
#include <nvds_obj_encode.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <librdkafka/rdkafkacpp.h>
#include <algorithm>
#include <cmath>
#include "message.h"
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

static void generate_ts_rfc3339(char *buf, int buf_size)
{
    time_t tloc;
    struct tm tm_log;
    struct timespec ts;
    char strmsec[6]; //.nnnZ\0

    clock_gettime(CLOCK_REALTIME, &ts);
    memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
    gmtime_r(&tloc, &tm_log);
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    int ms = ts.tv_nsec / 1000000;
    g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
    strncat(buf, strmsec, buf_size);
}

static void
generate_event_msg_meta(gpointer data, NvDsObjectMeta *obj_meta)
{
    NvDsFaceMetaData *faceMeta = NULL;
    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
        {
            faceMeta = reinterpret_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);
        }
    }
    NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *)data;
    meta->bbox.top = obj_meta->detector_bbox_info.org_bbox_coords.top;
    meta->bbox.left = obj_meta->detector_bbox_info.org_bbox_coords.left;
    meta->bbox.width = obj_meta->detector_bbox_info.org_bbox_coords.width;
    meta->bbox.height = obj_meta->detector_bbox_info.org_bbox_coords.height;

    meta->ts = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN + 1);
    meta->objectId = (gchar *)g_malloc0(MAX_LABEL_SIZE);

    strncpy(meta->objectId, obj_meta->obj_label, MAX_LABEL_SIZE);
    generate_ts_rfc3339(meta->ts, MAX_TIME_STAMP_LEN);
    if (obj_meta->class_id == FACE_CLASS_ID)
    {
        FaceEventMsgData *obj = (FaceEventMsgData *)g_malloc0(sizeof(FaceEventMsgData));

        obj->feature = g_strdup((gchar *)b64encode(faceMeta->feature, FEATURE_SIZE));
        meta->objType = NVDS_OBJECT_TYPE_FACE;
        meta->extMsg = obj;
        meta->extMsgSize = sizeof(FaceEventMsgData);
    }
}

static gpointer meta_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;
    dstMeta = (NvDsEventMsgMeta *)g_memdup(srcMeta, sizeof(NvDsEventMsgMeta));
    if (srcMeta->ts)
        dstMeta->ts = g_strdup(srcMeta->ts);

    if (srcMeta->objectId)
    {
        dstMeta->objectId = g_strdup(srcMeta->objectId);
    }
    if (srcMeta->extMsgSize > 0)
    {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_FACE)
        {
            FaceEventMsgData *srcObj = (FaceEventMsgData *)srcMeta->extMsg;
            FaceEventMsgData *obj = (FaceEventMsgData *)g_malloc0(sizeof(FaceEventMsgData));
            if (srcObj->feature)
            {
                obj->feature = g_strdup(srcObj->feature);
            }
            dstMeta->extMsg = obj;
            dstMeta->extMsgSize = sizeof(FaceEventMsgData);
        }
    }
    return dstMeta;
}

static void meta_free_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    g_free(srcMeta->ts);
    if (srcMeta->objectId)
    {
        g_free(srcMeta->objectId);
    }
    if (srcMeta->extMsgSize > 0)
    {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_FACE)
        {
            FaceEventMsgData *obj = (FaceEventMsgData *)srcMeta->extMsg;
            if (obj->feature)
            {
                g_free(obj->feature);
            }
        }
        g_free(srcMeta->extMsg);
        srcMeta->extMsgSize = 0;
    }
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}

GstPadProbeReturn NvInferFaceBin::osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GST_ASSERT(buf);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    GST_ASSERT(batch_meta);
    GstElement *tiler = reinterpret_cast<GstElement *>(_udata);
    GST_ASSERT(tiler);
    gint tiler_rows, tiler_cols, tiler_width, tiler_height;
    g_object_get(tiler, "rows", &tiler_rows, "columns", &tiler_cols, "width", &tiler_width, "height", &tiler_height, NULL);

    guint num_rects = 0;
    // loop through each frame in batch
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsMetaList *l_user = NULL;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        float muxer_output_height = frame_meta->pipeline_height;
        float muxer_output_width = frame_meta->pipeline_width;
        // translate from batch_id to the position of this frame in tiler
        int tiler_col = frame_meta->batch_id / tiler_cols;
        int tiler_row = frame_meta->batch_id % tiler_cols;
        int offset_x = tiler_col * tiler_width / tiler_cols;
        int offset_y = tiler_row * tiler_height / tiler_rows;
        // loop through each object in frame data

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
            if (obj_meta->class_id == FACE_CLASS_ID)
            {
                num_rects++;
                // draw landmark here
                for (l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
                {
                    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
                    if (user_meta->base_meta.meta_type != (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
                    {
                        continue;
                    }
                    NvDsFaceMetaData *faceMeta = static_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);
                    NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                    display_meta->num_circles = NUM_FACEMARK;
                    for (int j = 0; j < NUM_FACEMARK; j++)
                    {
                        float x = faceMeta->faceMark[2 * j] * tiler_width / muxer_output_width;
                        float y = faceMeta->faceMark[2 * j + 1] * tiler_height / muxer_output_height;
                        display_meta->circle_params[j].xc = static_cast<unsigned int>(x);
                        display_meta->circle_params[j].yc = static_cast<unsigned int>(y);
                        display_meta->circle_params[j].radius = 4;
                        display_meta->circle_params[j].circle_color.red = 1.0;
                        display_meta->circle_params[j].circle_color.green = 1.0;
                        display_meta->circle_params[j].circle_color.blue = 0.0;
                        display_meta->circle_params[j].circle_color.alpha = 1.0;
                        display_meta->circle_params[j].has_bg_color = 1;
                        display_meta->circle_params[j].bg_color.red = 1.0;
                        display_meta->circle_params[j].bg_color.green = 0.0;
                        display_meta->circle_params[j].bg_color.blue = 0.0;
                        display_meta->circle_params[j].bg_color.alpha = 1.0;
                    }
                    nvds_add_display_meta_to_frame(frame_meta, display_meta);
                }

                // ================== EVENT MESSAGE DATA ========================
                NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));

                generate_event_msg_meta(msg_meta, obj_meta);

                NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool(batch_meta);
                if (user_event_meta)
                {
                    user_event_meta->user_meta_data = (void *)msg_meta;
                    user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
                    user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc)meta_copy_func;
                    user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc)meta_free_func;
                    nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
                }
                else
                {
                    g_print("Error in attaching event meta to buffer\n");
                }
            }
        }

        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        display_meta->num_labels = 1;
        NvOSD_TextParams *txt_params = &display_meta->text_params[0];
        txt_params->display_text = reinterpret_cast<char *>(g_malloc0(MAX_DISPLAY_LEN));
        int offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number = %d Number of faces = %d", frame_meta->frame_num, num_rects);
        // g_print("Frame Number = %d Number of faces = %d\n", frame_meta->frame_num, num_rects);
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }
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

            /* Iterate final rectangules and attach result into frame's obj_meta_list */
            for (const auto &obj : res)
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
    nvds_obj_enc_finish((NvDsObjEncCtxHandle)_udata);

    return GST_PAD_PROBE_OK;
}

static size_t WriteJsonCallback(char *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

static gpointer XFace_msg_meta_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = (NvDsEventMsgMeta *)g_memdup(srcMeta, sizeof(NvDsEventMsgMeta));
    dstMeta->extMsg = g_malloc0(sizeof(XFaceMetaMsg));
    XFaceMetaMsg *srcExtMsg = (XFaceMetaMsg *)srcMeta->extMsg;
    XFaceMetaMsg *dstExtMsg = (XFaceMetaMsg *)dstMeta->extMsg;

    dstMeta->componentId = srcMeta->componentId;
    dstExtMsg->cameraId = g_strdup(srcExtMsg->cameraId);
    dstExtMsg->sessionId = g_strdup(srcExtMsg->sessionId);
    dstExtMsg->frameId = srcExtMsg->frameId;
    dstExtMsg->timestamp = srcExtMsg->timestamp;
    dstExtMsg->num_face_obj = srcExtMsg->num_face_obj;
    dstExtMsg->num_mot_obj = srcExtMsg->num_mot_obj;

    dstExtMsg->mot_meta_list = (NvDsMOTMsgData **)g_malloc0(dstExtMsg->num_mot_obj * sizeof(NvDsMOTMsgData *));
    dstExtMsg->face_meta_list = (NvDsFaceMsgData **)g_malloc0(dstExtMsg->num_face_obj * sizeof(NvDsFaceMsgData *));

    // Copy Face
    for (int i = 0; i < dstExtMsg->num_face_obj; i++)
    {
        NvDsFaceMsgData *msg_sub_meta = (NvDsFaceMsgData *)g_malloc0(sizeof(NvDsFaceMsgData));
        msg_sub_meta->bbox.top = srcExtMsg->face_meta_list[i]->bbox.top;
        msg_sub_meta->bbox.left = srcExtMsg->face_meta_list[i]->bbox.left;
        msg_sub_meta->bbox.width = srcExtMsg->face_meta_list[i]->bbox.width;
        msg_sub_meta->bbox.height = srcExtMsg->face_meta_list[i]->bbox.height;

        msg_sub_meta->confidence_score = srcExtMsg->face_meta_list[i]->confidence_score; // confidence score
        msg_sub_meta->feature = g_strdup(srcExtMsg->face_meta_list[i]->feature);         // face feature
        msg_sub_meta->staff_id = g_strdup(srcExtMsg->face_meta_list[i]->staff_id);       // Staff_code
        msg_sub_meta->name = g_strdup(srcExtMsg->face_meta_list[i]->name);               // Staff_code
        msg_sub_meta->encoded_img = g_strdup(srcExtMsg->face_meta_list[i]->encoded_img);

        dstExtMsg->face_meta_list[i] = msg_sub_meta;
    }

    // Copy MOT
    for (int i = 0; i < dstExtMsg->num_mot_obj; i++)
    {
        NvDsMOTMsgData *msg_sub_meta = (NvDsMOTMsgData *)g_malloc0(sizeof(NvDsMOTMsgData));
        msg_sub_meta->bbox.top = srcExtMsg->mot_meta_list[i]->bbox.top;
        msg_sub_meta->bbox.left = srcExtMsg->mot_meta_list[i]->bbox.left;
        msg_sub_meta->bbox.width = srcExtMsg->mot_meta_list[i]->bbox.width;
        msg_sub_meta->bbox.height = srcExtMsg->mot_meta_list[i]->bbox.height;
        msg_sub_meta->track_id = srcExtMsg->mot_meta_list[i]->track_id;

        msg_sub_meta->embedding = g_strdup(srcExtMsg->mot_meta_list[i]->embedding); // person embedding

        dstExtMsg->mot_meta_list[i] = msg_sub_meta;
    }

    dstMeta->extMsgSize = srcMeta->extMsgSize;
    return dstMeta;
}

static void XFace_msg_meta_release_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

    if (srcMeta->extMsgSize > 0)
    {
        // free extMsg content
        XFaceMetaMsg *srcExtMsg = (XFaceMetaMsg *)srcMeta->extMsg;
        // g_free(srcExtMsg->timestamp);
        g_free(srcExtMsg->cameraId);
        g_free(srcExtMsg->sessionId);
        // Delete face

        for (int i = 0; i < srcExtMsg->num_face_obj; i++)
        {
            NvDsFaceMsgData *msg_sub_meta = srcExtMsg->face_meta_list[i];

            g_free(msg_sub_meta->feature);
            g_free(msg_sub_meta->staff_id);
            g_free(msg_sub_meta->name);
            g_free(msg_sub_meta->encoded_img);

            g_free(msg_sub_meta);
        }
        g_free(srcExtMsg->face_meta_list);
        srcExtMsg->num_face_obj = 0;

        // Delete MOT
        for (int i = 0; i < srcExtMsg->num_mot_obj; i++)
        {
            NvDsMOTMsgData *msg_sub_meta = srcExtMsg->mot_meta_list[i];
            g_free(msg_sub_meta->embedding);
            g_free(msg_sub_meta);
        }
        g_free(srcExtMsg->mot_meta_list);
        srcExtMsg->num_mot_obj = 0;
        // free extMsg
        g_free(srcMeta->extMsg);
        // free extMsgSize
        srcMeta->extMsgSize = 0;
    }
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}

void getFaceMetaData(NvDsFrameMeta *frame_meta, NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta, std::vector<NvDsFaceMsgData *> &face_meta_list,
                     user_callback_data *callback_data, NvDsInferLayerInfo *output_layer_info)
{
    NvDsFaceMsgData *face_msg_sub_meta = (NvDsFaceMsgData *)g_malloc0(sizeof(NvDsFaceMsgData));
    face_msg_sub_meta->bbox.top = obj_meta->rect_params.top;
    face_msg_sub_meta->bbox.left = obj_meta->rect_params.left;
    face_msg_sub_meta->bbox.width = obj_meta->rect_params.width;
    face_msg_sub_meta->bbox.height = obj_meta->rect_params.height;

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

            // Send HTTP request
            CURL *curl = callback_data->curl;
            std::string response_string;
            const char *data = gen_body(1, b64encode(cur_feature, FEATURE_SIZE));
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteJsonCallback);

            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

            // request over HTTP/2, using the same connection!
            CURLcode res = curl_easy_perform(curl);

            if (res != CURLE_OK)
            {
                face_msg_sub_meta->confidence_score = 0;
                face_msg_sub_meta->staff_id = g_strdup("000000");
                face_msg_sub_meta->name = g_strdup("Unknown");
            }
            else
            {
                QDTLog::info("Response string = {}", response_string);
                std::string response_json = response_string.substr(1, response_string.size() - 2);
                Document doc;
                doc.Parse(response_json.c_str());
                Value &s = doc["score"];
                face_msg_sub_meta->confidence_score = s.GetDouble();
                s = doc["code"];
                face_msg_sub_meta->staff_id = g_strdup(s.GetString());
                s = doc["name"];
                face_msg_sub_meta->name = g_strdup(s.GetString());
                if (std::string(face_msg_sub_meta->name) != std::string("Unknown") && face_msg_sub_meta->confidence_score > callback_data->face_feature_confidence_threshold)
                {
                    obj_meta->text_params.x_offset = obj_meta->rect_params.left;
                    obj_meta->text_params.y_offset = std::max(0.0f, obj_meta->rect_params.top - 10);
                    obj_meta->text_params.display_text = (char *)g_malloc0(64 * sizeof(char));
                    snprintf(obj_meta->text_params.display_text, 64, face_msg_sub_meta->name, obj_meta->object_id);
                    obj_meta->text_params.font_params.font_name = (char *)"Serif";
                    obj_meta->text_params.font_params.font_size = 10;
                    obj_meta->text_params.font_params.font_color = {1.0, 1.0, 1.0, 1.0};
                    obj_meta->text_params.set_bg_clr = 1;
                    obj_meta->text_params.text_bg_clr = {0.0, 0.0, 0.0, 1.0};
                    nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
                }
            }
        }
        else if (user_meta->base_meta.meta_type == NVDS_CROP_IMAGE_META)
        {
            NvDsObjEncOutParams *enc_jpeg_image =
                (NvDsObjEncOutParams *)user_meta->user_meta_data;
            face_msg_sub_meta->encoded_img = g_strdup(b64encode(enc_jpeg_image->outBuffer, enc_jpeg_image->outLen));
        }
    }

    face_meta_list.push_back(face_msg_sub_meta);

    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        FILE *file;

        if (user_meta->base_meta.meta_type == NVDS_CROP_IMAGE_META)
        {
            if (face_msg_sub_meta->confidence_score > callback_data->face_feature_confidence_threshold && std::string(face_msg_sub_meta->name) != std::string("Unknown"))
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

void getMOTMetaData(NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta, std::vector<NvDsMOTMsgData *> &mot_meta_list)
{
    NvDsMOTMsgData *mot_msg_sub_meta = (NvDsMOTMsgData *)g_malloc0(sizeof(NvDsMOTMsgData));
    mot_msg_sub_meta->bbox.top = obj_meta->rect_params.top;
    mot_msg_sub_meta->bbox.left = obj_meta->rect_params.left;
    mot_msg_sub_meta->bbox.width = obj_meta->rect_params.width;
    mot_msg_sub_meta->bbox.height = obj_meta->rect_params.height;
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

    const auto p1 = std::chrono::system_clock::now();
    double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch()).count();
    callback_data->timestamp = timestamp;
    /* Assign feature to NvDsFaceMetaData */
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);

        std::vector<NvDsFaceMsgData *> face_sub_meta_list;
        std::vector<NvDsMOTMsgData *> mot_sub_meta_list;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
            if (obj_meta->class_id == FACE_CLASS_ID)
            {
                getFaceMetaData(frame_meta, batch_meta, obj_meta, face_sub_meta_list, callback_data, output_layer_info);
            }
            else if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
            {
                getMOTMetaData(batch_meta, obj_meta, mot_sub_meta_list);
            }
        }

        // ===================================== XFace MetaData sent to Kafka =====================================
        XFaceMetaMsg *msg_meta_content = (XFaceMetaMsg *)g_malloc0(sizeof(XFaceMetaMsg));
        // Get MOT meta
        msg_meta_content->num_mot_obj = mot_sub_meta_list.size();
        msg_meta_content->mot_meta_list = (NvDsMOTMsgData **)g_malloc0(mot_sub_meta_list.size() * sizeof(NvDsMOTMsgData *));
        memcpy(msg_meta_content->mot_meta_list, mot_sub_meta_list.data(), mot_sub_meta_list.size() * sizeof(NvDsMOTMsgData *));

        // Get Face meta
        msg_meta_content->num_face_obj = face_sub_meta_list.size();
        msg_meta_content->face_meta_list = (NvDsFaceMsgData **)g_malloc0(face_sub_meta_list.size() * sizeof(NvDsFaceMsgData *));
        memcpy(msg_meta_content->face_meta_list, face_sub_meta_list.data(), face_sub_meta_list.size() * sizeof(NvDsFaceMsgData *));

        // Generate timestamp
        msg_meta_content->timestamp = timestamp;
        msg_meta_content->cameraId = g_strdup(std::string(callback_data->video_name[frame_meta->source_id]).c_str());
        msg_meta_content->frameId = frame_meta->frame_num;
        msg_meta_content->sessionId = g_strdup(callback_data->session_id);

        // This is where to create the final NvDsEventMsgMeta before sending
        NvDsEventMsgMeta *meta_msg = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
        meta_msg->extMsg = (void *)msg_meta_content;
        meta_msg->extMsgSize = sizeof(XFaceMetaMsg);
        meta_msg->componentId = 1;

        gchar *message = generate_XFaceRawMeta_message(meta_msg);
        RdKafka::ErrorCode err = callback_data->kafka_producer->producer->produce(callback_data->metadata_topic,
                                                                                  RdKafka::Topic::PARTITION_UA,
                                                                                  RdKafka::Producer::RK_MSG_COPY,
                                                                                  (gchar *)message,
                                                                                  std::string(message).length(),
                                                                                  NULL, 0,
                                                                                  0, NULL, NULL);
        if (err != RdKafka::ERR_NO_ERROR)
        {
            QDTLog::error("{} Failed to produce to topic", RdKafka::err2str(err));

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
    NvBufSurfaceUnMap(surface, -1, -1);
    gst_buffer_unmap(buf, &in_map_info);
}

GstPadProbeReturn NvInferFaceBin::tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GST_ASSERT(buf);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    GST_ASSERT(batch_meta);

    GstElement *tiler = reinterpret_cast<GstElement *>(_udata);
    GST_ASSERT(tiler);
    gint tiler_rows, tiler_cols, tiler_width, tiler_height = 0;
    g_object_get(tiler, "rows", &tiler_rows, "columns", &tiler_cols, "width", &tiler_width, "height", &tiler_height, NULL);
    assert(tiler_height != 0);

    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsMetaList *l_user = NULL;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        float muxer_output_height = frame_meta->pipeline_height;
        float muxer_output_width = frame_meta->pipeline_width;
        // translate from batch_id to the position of this frame in tiler
        int tiler_col = frame_meta->batch_id % tiler_cols;
        int tiler_row = frame_meta->batch_id / tiler_cols;
        int offset_x = tiler_col * tiler_width / tiler_cols;
        int offset_y = tiler_row * tiler_height / tiler_rows;
        // g_print("in tiler_sink_pad_buffer_probe batch_id = %d, the tiler offset = %d, %d\n", frame_meta->batch_id, offset_x, offset_y);
        // loop through each object in frame data
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
            if (obj_meta->class_id == FACE_CLASS_ID)
            {
                for (l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
                {
                    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
                    if (user_meta->base_meta.meta_type != (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
                    {
                        continue;
                    }

                    NvDsFaceMetaData *faceMeta = static_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);
                    // scale the landmark data base on tiler
                    for (int j = 0; j < NUM_FACEMARK; j++)
                    {
                        faceMeta->faceMark[2 * j] = faceMeta->faceMark[2 * j] / tiler_cols + offset_x;
                        faceMeta->faceMark[2 * j + 1] = faceMeta->faceMark[2 * j + 1] / tiler_rows + offset_y;
                    }
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
}
