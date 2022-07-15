#include "BufferProbe.h"

// static GstPadProbeReturn
// osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
// {
//     GstBuffer *buf = (GstBuffer *)info->data;
//     guint num_rects = 0;
//     NvDsObjectMeta *obj_meta = NULL;
//     guint vehicle_count = 0;
//     guint person_count = 0;
//     NvDsMetaList *l_frame = NULL;
//     NvDsMetaList *l_obj = NULL;
//     NvDsDisplayMeta *display_meta = NULL;

//     NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

//     for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
//     {
//         NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
//         int offset = 0;
//         for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
//         {
//             obj_meta = (NvDsObjectMeta *)(l_obj->data);
//             if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
//             {
//                 vehicle_count++;
//                 num_rects++;
//             }
//             if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
//             {
//                 person_count++;
//                 num_rects++;
//             }
//         }

//         display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
//         NvOSD_TextParams *txt_params = &display_meta->text_params[0];
//         display_meta->num_labels = 1;
//         txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
//         offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
//         offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

//         /* Now set the offsets where the string should appear */
//         txt_params->x_offset = 10;
//         txt_params->y_offset = 12;

//         /* Font , font-color and font-size */
//         txt_params->font_params.font_name = "Serif";
//         txt_params->font_params.font_size = 10;
//         txt_params->font_params.font_color.red = 1.0;
//         txt_params->font_params.font_color.green = 1.0;
//         txt_params->font_params.font_color.blue = 1.0;
//         txt_params->font_params.font_color.alpha = 1.0;

//         /* Text background color */
//         txt_params->set_bg_clr = 1;
//         txt_params->text_bg_clr.red = 0.0;
//         txt_params->text_bg_clr.green = 0.0;
//         txt_params->text_bg_clr.blue = 0.0;
//         txt_params->text_bg_clr.alpha = 1.0;

//         nvds_add_display_meta_to_frame(frame_meta, display_meta);
//     }

//     // g_print("Frame Number = %d Number of objects = %d "
//     //         "Vehicle Count = %d Person Count = %d\n",
//     //         frame_number, num_rects, vehicle_count, person_count);
//     // frame_number++;
//     return GST_PAD_PROBE_OK;
// }