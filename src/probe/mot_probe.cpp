#include "probe.h"

#ifdef ENABLE_EVAL_SPEED
bool START_MEASURE_FPS = false;
auto LAST_TICK = std::chrono::system_clock::now();
double TOTAL_TIME = 0;
double NUM_TICKS = 0;
#endif


static gpointer sgie_src_pad_meta_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = (NvDsEventMsgMeta*) g_memdup (srcMeta, sizeof(NvDsEventMsgMeta));

    // copy extMsg
	dstMeta->extMsg = g_malloc0(sizeof(EventMsgSubMeta));
    EventMsgSubMeta* srcExtMsg = (EventMsgSubMeta*) srcMeta->extMsg;
    EventMsgSubMeta* dstExtMsg = (EventMsgSubMeta*) dstMeta->extMsg;

    // copy extMsg -- copy main information
    dstExtMsg->type = srcExtMsg->type;
    dstExtMsg->frameId = srcExtMsg->frameId;
    dstExtMsg->sensorId = srcExtMsg->sensorId;
    dstExtMsg->num_msg_sub_meta = srcExtMsg->num_msg_sub_meta;
    
    // copy extMsg -- copy msg_sub_meta_list
    dstExtMsg->msg_sub_meta_list = (NvDsEventMsgMeta**) g_malloc0(
        dstExtMsg->num_msg_sub_meta * sizeof(NvDsEventMsgMeta*)
    );
    for (int i = 0; i < dstExtMsg->num_msg_sub_meta; i++) {
        NvDsEventMsgMeta *msg_sub_meta = (NvDsEventMsgMeta*) g_malloc0(sizeof(NvDsEventMsgMeta));
        msg_sub_meta->bbox.top = srcExtMsg->msg_sub_meta_list[i]->bbox.top;
        msg_sub_meta->bbox.left = srcExtMsg->msg_sub_meta_list[i]->bbox.left;
        msg_sub_meta->bbox.width = srcExtMsg->msg_sub_meta_list[i]->bbox.width;
        msg_sub_meta->bbox.height = srcExtMsg->msg_sub_meta_list[i]->bbox.height;

        msg_sub_meta->trackingId = srcExtMsg->msg_sub_meta_list[i]->trackingId;

        msg_sub_meta->frameId = srcExtMsg->msg_sub_meta_list[i]->frameId;
        msg_sub_meta->sensorId = srcExtMsg->msg_sub_meta_list[i]->sensorId;

        msg_sub_meta->objSignature.size = EMBEDDING_DIMS;
        msg_sub_meta->objSignature.signature = (double*) g_malloc0(
            EMBEDDING_DIMS * sizeof(double)
        );
        memcpy(msg_sub_meta->objSignature.signature, 
               srcExtMsg->msg_sub_meta_list[i]->objSignature.signature, 
               EMBEDDING_DIMS * sizeof(double));

        dstExtMsg->msg_sub_meta_list[i] = msg_sub_meta;
    }

    // copy extMsgSize
    dstMeta->extMsgSize = srcMeta->extMsgSize;

    return dstMeta;
}

static void sgie_src_pad_meta_free_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

    if (srcMeta->extMsgSize > 0) {
        // free extMsg content
        EventMsgSubMeta* srcExtMsg = (EventMsgSubMeta*) srcMeta->extMsg;
        for (int i = 0; i < srcMeta->extMsgSize; i++) {
            NvDsEventMsgMeta *msg_sub_meta = srcExtMsg->msg_sub_meta_list[i];
            g_free (msg_sub_meta->objSignature.signature);
            msg_sub_meta->objSignature.size = 0;
            g_free (msg_sub_meta);
        }
        g_free (srcExtMsg->msg_sub_meta_list);
        srcExtMsg->num_msg_sub_meta = 0;

        // free extMsg
        g_free (srcMeta->extMsg);

        // free extMsgSize
        srcMeta->extMsgSize = 0;
    }

    g_free (user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}


void parse_embedding_from_user_meta_data(
    NvDsUserMeta* user_meta, float*& embedding_data_f
) {
    NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta*) user_meta->user_meta_data;
    NvDsInferLayerInfo layer_info = tensor_meta->output_layers_info[0];
    embedding_data_f = (float*) tensor_meta->out_buf_ptrs_host[0];
}

void parse_detections_from_frame_meta(DETECTIONS &detections, NvDsFrameMeta *frame_meta) {
    NvDsObjectMetaList *l_object = frame_meta->obj_meta_list;
    while (l_object) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta*) l_object->data;

        NvDsUserMetaList *l_user = obj_meta->obj_user_meta_list;
        while (l_user) {
            NvDsUserMeta *user_meta = (NvDsUserMeta*) l_user->data;

            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                float *embedding_data_f;
                parse_embedding_from_user_meta_data(user_meta, embedding_data_f);
                DETECTION_ROW row;
                row.class_num = 0;
                row.confidence = obj_meta->confidence;
                row.tlwh = DETECTBOX(
                    obj_meta->rect_params.top, obj_meta->rect_params.left,
                    obj_meta->rect_params.width, obj_meta->rect_params.height
                );
                row.feature = FEATURE(embedding_data_f);
                detections.push_back(row);
            }

            l_user = l_user->next;
        }

        l_object = l_object->next;
    }
}


void make_obj_meta_from_track_box(NvDsObjectMeta *obj_meta, Track track) {
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
    obj_meta->text_params.display_text = (char*) g_malloc0(64 * sizeof(char));
    snprintf(obj_meta->text_params.display_text, 64, "PersonBox_%lu", obj_meta->object_id);
    obj_meta->text_params.font_params.font_name = (char*) "Serif";
    obj_meta->text_params.font_params.font_size = 10;
    obj_meta->text_params.font_params.font_color = {1.0, 1.0, 1.0, 1.0};
    obj_meta->text_params.set_bg_clr = 1;
    obj_meta->text_params.text_bg_clr = {0.0, 0.0, 0.0, 1.0};
}


void make_msg_sub_meta(Track track, NvDsFrameMeta* frame_meta, NvDsEventMsgMeta*& msg_sub_meta) {
    DETECTBOX track_box = track.to_tlwh();

    // TODO: Free memory of this pointer later
    NvDsEventMsgMeta *_msg_sub_meta = (NvDsEventMsgMeta*) g_malloc0(sizeof(NvDsEventMsgMeta));

    // Object information
    _msg_sub_meta->bbox.top = track_box(0);
    _msg_sub_meta->bbox.left = track_box(1);
    _msg_sub_meta->bbox.width = track_box(2);
    _msg_sub_meta->bbox.height = track_box(3);
    _msg_sub_meta->trackingId = track.track_id;

    // Object embedding
    double *embedding_data = (double*) g_malloc0(EMBEDDING_DIMS * sizeof(double));
    FEATURE last_feature = track.last_feature;
    Eigen::Matrix<
        double, 1, EMBEDDING_DIMS, Eigen::RowMajor
    > last_feature_d = last_feature.cast<double>();
    Eigen::Map<
        Eigen::Matrix<double, 1, EMBEDDING_DIMS, Eigen::RowMajor>
    >(embedding_data, last_feature_d.rows(), last_feature_d.cols()) = last_feature_d;
    _msg_sub_meta->objSignature.signature = embedding_data;
    _msg_sub_meta->objSignature.size = EMBEDDING_DIMS;

    // Frame information
    _msg_sub_meta->frameId = frame_meta->frame_num;
    _msg_sub_meta->sensorId = frame_meta->source_id;

    msg_sub_meta = _msg_sub_meta;
}


/**
 * @brief  Get object embeddings, carry out tracking, and make EventMsgSubMeta
 *
 * @param frame_meta
 * @param msg_meta_content
 */
void sgie_src_pad_make_msg_sub_meta_list(
    std::vector<Track> tracks, NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta, EventMsgSubMeta*& msg_meta_content
) {
    std::vector<NvDsEventMsgMeta*> _msg_sub_meta_list;

    // make NvDsEventMsgMeta
    nvds_clear_obj_meta_list(frame_meta, frame_meta->obj_meta_list);
    for (Track &track : tracks) {
        if(!track.is_confirmed() || track.time_since_update > 1) continue;

        // Make NvDsObjectMeta
        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
        make_obj_meta_from_track_box(obj_meta, track);
        nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);

        // Make NvDsEventMsgMeta
        NvDsEventMsgMeta *msg_sub_meta;
        make_msg_sub_meta(track, frame_meta, msg_sub_meta);
        _msg_sub_meta_list.push_back(msg_sub_meta);

#ifdef ENABLE_EVAL_MOT
        // Save MOT results for evaluation
        char *filename = (char*) g_malloc0(64 * sizeof(char));
        snprintf(filename, 64, "stream_%u.txt", frame_meta->source_id);
        std::string save_path = std::string(MOT_EVAL_SAVE_DIR) + "/" + std::string(filename);

        float h_ratio = ((float) frame_meta->source_frame_height) / ((float) frame_meta->pipeline_height);
        float w_ratio = ((float) frame_meta->source_frame_width) / ((float) frame_meta->pipeline_width);
        float obj_left = obj_meta->rect_params.left * w_ratio;
        float obj_top = obj_meta->rect_params.top * h_ratio;
        float obj_width = obj_meta->rect_params.width * w_ratio;
        float obj_height = obj_meta->rect_params.height * h_ratio;

        std::ofstream outfile;
        outfile.open(save_path, std::ios::out | std::ios::app);
        outfile << frame_meta->frame_num << "," << obj_meta->object_id << ","
                << (int) obj_left << "," << (int) obj_top << ","
                << (int) obj_width << "," << (int) obj_height << ","
                << "-1,-1,-1,-1\n";
        outfile.close();
#endif
    }
    
    // make EventMsgSubMeta
    msg_meta_content = (EventMsgSubMeta*) g_malloc0(sizeof(EventMsgSubMeta));
    msg_meta_content->type = EventMsgSubMetaType::SGIE_EVENT;
    msg_meta_content->frameId = frame_meta->frame_num;
    msg_meta_content->sensorId = frame_meta->source_id;
    msg_meta_content->num_msg_sub_meta = _msg_sub_meta_list.size();
    msg_meta_content->msg_sub_meta_list = (NvDsEventMsgMeta**) g_malloc0(
        _msg_sub_meta_list.size() * sizeof(NvDsEventMsgMeta*)
    );
    memcpy(msg_meta_content->msg_sub_meta_list, _msg_sub_meta_list.data(),
           _msg_sub_meta_list.size() * sizeof(NvDsEventMsgMeta*));
}

GstPadProbeReturn 
sgie_mot_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    MOTTrackerList *tracker_list = (MOTTrackerList*) user_data;

    GstBuffer *gst_buffer = gst_pad_probe_info_get_buffer(info);
    if (!gst_buffer) {
        gst_print ("no GstBuffer found in sgie_mot_src_pad_buffer_probe()\n");
        gst_object_unref(gst_buffer);
        return GST_PAD_PROBE_OK;
    }
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(gst_buffer);
    NvDsFrameMetaList *l_frame = batch_meta->frame_meta_list;
    while (l_frame) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta*) l_frame->data;
        tracker *_tracker = tracker_list->trackers + frame_meta->source_id;

        // Track with DeepSORT
        DETECTIONS detections;
        parse_detections_from_frame_meta(detections, frame_meta);
        _tracker->predict();
        _tracker->update(detections);

        // Make NvDsObjectMeta and EventMsgSubMeta
        EventMsgSubMeta *msg_meta_content;
        sgie_src_pad_make_msg_sub_meta_list(
            _tracker->tracks, batch_meta, frame_meta, msg_meta_content
        );

        // Make NvDsEventMsgMeta obj
        NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta*) g_malloc0(sizeof(NvDsEventMsgMeta));
        msg_meta->extMsg = (void*) msg_meta_content;
        msg_meta->extMsgSize = msg_meta_content->num_msg_sub_meta;
        
        // Pack EventMsgMeta into UserMeta
        NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool (batch_meta);
        if (user_event_meta) {
            user_event_meta->user_meta_data = (void *) msg_meta;
            user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
            user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc) sgie_src_pad_meta_copy_func;
            user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc) sgie_src_pad_meta_free_func;
            nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
        }

        l_frame = l_frame->next;
    }

#ifdef ENABLE_EVAL_SPEED
    if (!START_MEASURE_FPS) {
        START_MEASURE_FPS = true;
        LAST_TICK = std::chrono::system_clock::now();
    } else {
        // Measure
        auto tick = std::chrono::system_clock::now();
        double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(tick - LAST_TICK).count();

        // Update
        LAST_TICK = tick;
        TOTAL_TIME += elapsed_time;
        NUM_TICKS++;

        // Statistics
        double AVG_RUNTIME = TOTAL_TIME / NUM_TICKS / 1e6;
        double AVG_FPS = 1.0 / AVG_RUNTIME;
        std::cout << "Average runtime: " << AVG_RUNTIME << " - Average FPS: "  << AVG_FPS << std::endl;
    }
#endif

    return GST_PAD_PROBE_OK;
}
