#include "MOTBin.h"
#include "QDTLog.h"
#include "utils.h"

void parse_embedding_from_user_meta_data(
    NvDsUserMeta *user_meta, float *&embedding_data_f)
{
    NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
    NvDsInferLayerInfo layer_info = tensor_meta->output_layers_info[0];
    embedding_data_f = (float *)tensor_meta->out_buf_ptrs_host[0];
}

static void parse_detections_from_frame_meta(DETECTIONS &detections, NvDsFrameMeta *frame_meta, float mot_confidence_threshold)
{
    NvDsObjectMetaList *l_object = frame_meta->obj_meta_list;
    while (l_object)
    {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_object->data;

        NvDsUserMetaList *l_user = obj_meta->obj_user_meta_list;
        while (l_user)
        {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;

            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
            {
                float *embedding_data_f;
                parse_embedding_from_user_meta_data(user_meta, embedding_data_f);
                DETECTION_ROW row;
                row.class_num = 0;
                row.confidence = obj_meta->confidence;

                if (row.confidence < mot_confidence_threshold)
                {
                    break;
                }
                row.tlwh = DETECTBOX(
                    obj_meta->rect_params.top, obj_meta->rect_params.left,
                    obj_meta->rect_params.width, obj_meta->rect_params.height);
                row.feature = FEATURE(embedding_data_f);
                detections.push_back(row);
            }
            l_user = l_user->next;
        }
        l_object = l_object->next;
    }
}

void make_obj_meta_from_track_box(NvDsObjectMeta *obj_meta, Track track)
{
    DETECTBOX track_box = track.to_tlwh();

    obj_meta->rect_params.top = track_box(0);
    obj_meta->rect_params.left = track_box(1);
    obj_meta->rect_params.width = track_box(2);
    obj_meta->rect_params.height = track_box(3);

    obj_meta->rect_params.has_bg_color = 0;
    obj_meta->rect_params.bg_color = {1.0, 1.0, 0.0, 0.4};
    obj_meta->rect_params.border_width = 3;
    obj_meta->rect_params.border_color = {1, 0, 0, 1};

    obj_meta->confidence = POST_TRACK_SCORE;
    obj_meta->class_id = PGIE_CLASS_ID_PERSON;
    obj_meta->object_id = track.track_id;
    strcpy(obj_meta->obj_label, "PersonBox");

    obj_meta->text_params.x_offset = obj_meta->rect_params.left;
    obj_meta->text_params.y_offset = std::max(0.0f, obj_meta->rect_params.top - 10);
    obj_meta->text_params.display_text = (char *)g_malloc0(64 * sizeof(char));
    snprintf(obj_meta->text_params.display_text, 64, "PersonBox_%lu", obj_meta->object_id);
    obj_meta->text_params.font_params.font_name = (char *)"Serif";
    obj_meta->text_params.font_params.font_size = 10;
    obj_meta->text_params.font_params.font_color = {1.0, 1.0, 1.0, 1.0};
    obj_meta->text_params.set_bg_clr = 1;
    obj_meta->text_params.text_bg_clr = {0.0, 0.0, 0.0, 1.0};
}


gpointer user_copy_mot_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(data);
    NvDsMOTMetaData *mot_meta_data_ptr = reinterpret_cast<NvDsMOTMetaData *>(
        user_meta->user_meta_data);
    NvDsMOTMetaData *new_mot_meta_data_ptr = reinterpret_cast<NvDsMOTMetaData *>(
        g_memdup(mot_meta_data_ptr, sizeof(NvDsMOTMetaData)));
    return reinterpret_cast<gpointer>(new_mot_meta_data_ptr);
}

/**
 * @brief implement NvDsMetaReleaseFun
 *
 * @param data
 * @param user_data
 */
void user_release_mot_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(data);
    NvDsMOTMetaData *mot_meta_data_ptr = reinterpret_cast<NvDsMOTMetaData *>(
        user_meta->user_meta_data);
    delete mot_meta_data_ptr;
}

GstPadProbeReturn NvInferMOTBin::sgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    user_callback_data * callback_data = (user_callback_data *)user_data;
    std::vector<std::shared_ptr<tracker>> trackers = callback_data->trackers;
    GstBuffer *gst_buffer = gst_pad_probe_info_get_buffer(info);
    NvDsMetaList *l_obj = NULL;

    if (!gst_buffer)
    {
        gst_print("no GstBuffer found in sgie_mot_src_pad_buffer_probe()\n");
        gst_object_unref(gst_buffer);
        return GST_PAD_PROBE_OK;
    }
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
    NvDsFrameMetaList *l_frame = batch_meta->frame_meta_list;

    while (l_frame)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        // Track with DeepSORT
        DETECTIONS detections;
        parse_detections_from_frame_meta(detections, frame_meta, callback_data->mot_confidence_threshold);
        trackers[frame_meta->source_id]->predict();
        trackers[frame_meta->source_id]->update(detections);

        nvds_clear_obj_meta_list(frame_meta, frame_meta->obj_meta_list);
        for (Track &track : trackers[frame_meta->source_id]->tracks)
        {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;

            // Create metadata for object including bbox, id and material for nvosd
            NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
            make_obj_meta_from_track_box(obj_meta, track);

            // Extract object embedding and add to user_meta_data of the obj_meta
            NvDsMOTMetaData *mot_meta_ptr = new NvDsMOTMetaData();

            float *embedding_data = (float *)g_malloc0(FEATURE_SIZE * sizeof(float));
            FEATURE last_feature = track.last_feature;
            // Eigen::Matrix<
            //     float, 1, FEATURE_SIZE, Eigen::RowMajor>
            //     last_feature_d = last_feature.cast<float>();
            Eigen::Map<
                Eigen::Matrix<float, 1, FEATURE_SIZE, Eigen::RowMajor>>(embedding_data, last_feature.rows(), last_feature.cols()) = last_feature;

            mot_meta_ptr->feature = g_strdup(b64encode((float *)embedding_data, FEATURE_SIZE));
            g_free(embedding_data);
            NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
            user_meta->user_meta_data = static_cast<void *>(mot_meta_ptr);
            user_meta->base_meta.meta_type = (NvDsMetaType)NVDS_OBJ_USER_META_MOT;
            user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)user_copy_mot_meta;
            user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)user_release_mot_meta;
            nvds_add_user_meta_to_obj(obj_meta, user_meta);
            nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
        }
        l_frame = l_frame->next;
    }

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn NvInferMOTBin::osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    return GST_PAD_PROBE_OK;
}
