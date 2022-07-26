#include "BufferProbe.h"

// GstPadProbeReturn osd_sink_pad_callback(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
// {
//     GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
//     GST_ASSERT(buf);
//     NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
//     GST_ASSERT(batch_meta);
//     GstElement *tiler = reinterpret_cast<GstElement *>(_udata);
//     GST_ASSERT(tiler);
//     gint tiler_rows, tiler_cols, tiler_width, tiler_height;
//     g_object_get(tiler, "rows", &tiler_rows, "columns", &tiler_cols, "width", &tiler_width, "height", &tiler_height, NULL);

//     for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
//         NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
//         float muxer_output_height = frame_meta->pipeline_height;
//         float muxer_output_width = frame_meta->pipeline_width;

//         // translate from batch_id to the position of this frame in tiler
//         int tiler_col = frame_meta->batch_id / tiler_cols;
//         int tiler_row = frame_meta->batch_id % tiler_cols;
//         int offset_x = tiler_col * tiler_width / tiler_cols;
//         int offset_y = tiler_row * tiler_height / tiler_rows;

//         guint num_person = 0;
//         guint num_faces = 0;

//         // loop through each object in frame data
//         for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
//             NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
//             if (FACE_CLASS_ID == obj_meta->class_id) {
//                 num_faces++;
//             } 
//             if (0 == obj_meta->class_id) {
//                 // label 0 in labels.txt
//                 num_person++;
//             }
//         }

//         // Set display text
//         NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
//         display_meta->num_labels = 1;
//         NvOSD_TextParams* nvosd_text_params = &display_meta->text_params[0];
//         nvosd_text_params->display_text = reinterpret_cast<char *>(g_malloc0(MAX_DISPLAY_LEN));
//         int offset = snprintf(nvosd_text_params->display_text, MAX_DISPLAY_LEN, "Frame Number = %d Persons = %d Faces = %d", frame_meta->frame_num, num_person, num_faces);
//         nvosd_text_params->x_offset = 10;
//         nvosd_text_params->y_offset = 12;
//         nvosd_text_params->font_params.font_name = const_cast<char*>("Serif");
//         nvosd_text_params->font_params.font_size = 10;
//         nvosd_text_params->font_params.font_color = {1.0, 1.0, 1.0, 1.0};
//         nvosd_text_params->set_bg_clr = 1;
//         nvosd_text_params->text_bg_clr = {0.0, 0.0, 0.0, 1.0};
//         nvds_add_display_meta_to_frame(frame_meta, display_meta);
//     }
    
//     return GST_PAD_PROBE_OK;
// }
