#include "NvInferBinBase.h"

GstPadProbeReturn NvInferBinBase::osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn NvInferBinBase::tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
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
