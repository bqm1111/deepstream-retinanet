#include "probe.h"
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

static gpointer meta_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = (NvDsEventMsgMeta *)g_memdup(srcMeta, sizeof(NvDsEventMsgMeta));

    if (srcMeta->ts)
        dstMeta->ts = g_strdup(srcMeta->ts);

    if (srcMeta->sensorStr)
        dstMeta->sensorStr = g_strdup(srcMeta->sensorStr);

    if (srcMeta->objSignature.size > 0)
    {
        dstMeta->objSignature.signature = (gdouble *)g_memdup(srcMeta->objSignature.signature,
                                                              srcMeta->objSignature.size);
        dstMeta->objSignature.size = srcMeta->objSignature.size;
    }

    if (srcMeta->objectId)
    {
        dstMeta->objectId = g_strdup(srcMeta->objectId);
    }

    if (srcMeta->extMsgSize > 0)
    {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE)
        {
            NvDsVehicleObject *srcObj = (NvDsVehicleObject *)srcMeta->extMsg;
            NvDsVehicleObject *obj = (NvDsVehicleObject *)g_malloc0(sizeof(NvDsVehicleObject));
            if (srcObj->type)
                obj->type = g_strdup(srcObj->type);
            if (srcObj->make)
                obj->make = g_strdup(srcObj->make);
            if (srcObj->model)
                obj->model = g_strdup(srcObj->model);
            if (srcObj->color)
                obj->color = g_strdup(srcObj->color);
            if (srcObj->license)
                obj->license = g_strdup(srcObj->license);
            if (srcObj->region)
                obj->region = g_strdup(srcObj->region);

            dstMeta->extMsg = obj;
            dstMeta->extMsgSize = sizeof(NvDsVehicleObject);
        }
        else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON)
        {
            NvDsPersonObject *srcObj = (NvDsPersonObject *)srcMeta->extMsg;
            NvDsPersonObject *obj = (NvDsPersonObject *)g_malloc0(sizeof(NvDsPersonObject));

            obj->age = srcObj->age;

            if (srcObj->gender)
                obj->gender = g_strdup(srcObj->gender);
            if (srcObj->cap)
                obj->cap = g_strdup(srcObj->cap);
            if (srcObj->hair)
                obj->hair = g_strdup(srcObj->hair);
            if (srcObj->apparel)
                obj->apparel = g_strdup(srcObj->apparel);
            dstMeta->extMsg = obj;
            dstMeta->extMsgSize = sizeof(NvDsPersonObject);
        }
    }

    return dstMeta;
}

static void meta_free_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

    g_free(srcMeta->ts);
    g_free(srcMeta->sensorStr);

    if (srcMeta->objSignature.size > 0)
    {
        g_free(srcMeta->objSignature.signature);
        srcMeta->objSignature.size = 0;
    }

    if (srcMeta->objectId)
    {
        g_free(srcMeta->objectId);
    }

    if (srcMeta->extMsgSize > 0)
    {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE)
        {
            NvDsVehicleObject *obj = (NvDsVehicleObject *)srcMeta->extMsg;
            if (obj->type)
                g_free(obj->type);
            if (obj->color)
                g_free(obj->color);
            if (obj->make)
                g_free(obj->make);
            if (obj->model)
                g_free(obj->model);
            if (obj->license)
                g_free(obj->license);
            if (obj->region)
                g_free(obj->region);
        }
        else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON)
        {
            NvDsPersonObject *obj = (NvDsPersonObject *)srcMeta->extMsg;

            if (obj->gender)
                g_free(obj->gender);
            if (obj->cap)
                g_free(obj->cap);
            if (obj->hair)
                g_free(obj->hair);
            if (obj->apparel)
                g_free(obj->apparel);
        }
        g_free(srcMeta->extMsg);
        srcMeta->extMsgSize = 0;
    }
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}

static void
generate_vehicle_meta(gpointer data)
{
    NvDsVehicleObject *obj = (NvDsVehicleObject *)data;

    obj->type = g_strdup("sedan");
    obj->color = g_strdup("blue");
    obj->make = g_strdup("Bugatti");
    obj->model = g_strdup("M");
    obj->license = g_strdup("XX1234");
    obj->region = g_strdup("CA");
}

static void
generate_person_meta(gpointer data)
{
    NvDsPersonObject *obj = (NvDsPersonObject *)data;
    obj->age = 45;
    obj->cap = g_strdup("none");
    obj->hair = g_strdup("black");
    obj->gender = g_strdup("male");
    obj->apparel = g_strdup("formal");
}

static void
generate_event_msg_meta(gpointer data, gint class_id, NvDsObjectMeta *obj_params)
{
    NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *)data;
    meta->sensorId = 0;
    meta->placeId = 0;
    meta->moduleId = 0;
    meta->sensorStr = g_strdup("sensor-0");
    meta->trackingId = 100;

    meta->ts = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN + 1);
    meta->objectId = (gchar *)g_malloc0(MAX_LABEL_SIZE);

    strncpy(meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);

    generate_ts_rfc3339(meta->ts, MAX_TIME_STAMP_LEN);

    /*
     * This demonstrates how to attach custom objects.
     * Any custom object as per requirement can be generated and attached
     * like NvDsVehicleObject / NvDsPersonObject. Then that object should
     * be handled in payload generator library (nvmsgconv.cpp) accordingly.
     */
    if (class_id == PGIE_CLASS_ID_VEHICLE)
    {
        meta->type = NVDS_EVENT_MOVING;
        meta->objType = NVDS_OBJECT_TYPE_VEHICLE;
        meta->objClassId = PGIE_CLASS_ID_VEHICLE;

        NvDsVehicleObject *obj = (NvDsVehicleObject *)g_malloc0(sizeof(NvDsVehicleObject));
        generate_vehicle_meta(obj);

        meta->extMsg = obj;
        meta->extMsgSize = sizeof(NvDsVehicleObject);
    }
    else if (class_id == PGIE_CLASS_ID_PERSON)
    {
        meta->type = NVDS_EVENT_ENTRY;
        meta->objType = NVDS_OBJECT_TYPE_PERSON;
        meta->objClassId = PGIE_CLASS_ID_PERSON;

        NvDsPersonObject *obj = (NvDsPersonObject *)g_malloc0(sizeof(NvDsPersonObject));
        generate_person_meta(obj);

        meta->extMsg = obj;
        meta->extMsgSize = sizeof(NvDsPersonObject);
    }
}

GstPadProbeReturn
osd_yolo_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        if (frame_meta == NULL)
        {
            continue;
        }
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if (obj_meta == NULL)
            {
                continue;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
            {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
            {
                person_count++;
                num_rects++;
            }

            // ================== DISPLAY ON OSD ========================
            display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            NvOSD_TextParams *txt_params = &display_meta->text_params[0];
            display_meta->num_labels = 1;
            txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
            offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
            offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

            /* Now set the offsets where the string should appear */
            txt_params->x_offset = 10;
            txt_params->y_offset = 12;

            /* Font , font-color and font-size */
            txt_params->font_params.font_name = "Serif";
            txt_params->font_params.font_size = 10;
            txt_params->font_params.font_color.red = 1.0;
            txt_params->font_params.font_color.green = 1.0;
            txt_params->font_params.font_color.blue = 1.0;
            txt_params->font_params.font_color.alpha = 1.0;

            /* Text background color */
            txt_params->set_bg_clr = 1;
            txt_params->text_bg_clr.red = 0.0;
            txt_params->text_bg_clr.green = 0.0;
            txt_params->text_bg_clr.blue = 0.0;
            txt_params->text_bg_clr.alpha = 1.0;

            nvds_add_display_meta_to_frame(frame_meta, display_meta);

            // ================== EVENT MESSAGE DATA ========================

            NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
            msg_meta->bbox.top = obj_meta->rect_params.top;
            msg_meta->bbox.left = obj_meta->rect_params.left;
            msg_meta->bbox.width = obj_meta->rect_params.width;
            msg_meta->bbox.height = obj_meta->rect_params.height;
            msg_meta->frameId = 10;
            msg_meta->trackingId = obj_meta->object_id;
            msg_meta->confidence = obj_meta->confidence;

            generate_event_msg_meta(msg_meta, obj_meta->class_id, obj_meta);
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
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn pgie_yolo_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    static NvDsInferNetworkInfo networkInfo{640, 640, 3};
    NvDsInferParseDetectionParams detectionParams;
    detectionParams.numClassesConfigured = 4;
    detectionParams.perClassPostclusterThreshold = {0.2, 0.2, 0.2, 0.2};
    static float groupThreshold = 1;
    static float groupEps = 0.2;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next)
        {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
            {
                continue;
            }
            NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
            for (unsigned int i = 0; i < meta->num_output_layers; i++)
            {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
                if (meta->out_buf_ptrs_dev[i])
                {
                    cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i], info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                }
            }
            std::vector<NvDsInferLayerInfo> outputLayersInfo(meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);
            std::vector<NvDsInferObjectDetectionInfo> objectList;
            // NvDsInferParseYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
        }
    }
    return GST_PAD_PROBE_OK;
}