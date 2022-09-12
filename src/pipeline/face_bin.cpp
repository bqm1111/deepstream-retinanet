#include "face_bin.h"
    struct feature_callback_data_t
    {
        int tensor_count = 0;
    };

    /**
     * @brief copy feature from device to host and assign to faceMeta->feature inplace.
     * NOTE: the network shape of nvfacealign and sgie must match, otherwise this function is called mulitple time with a same layers_infos.
     * 
     * @param buf 
     * @param network_info 
     * @param layers_info 
     * @param num_layers 
     * @param batch_size first dimension of the input tensor. 
     * @param user_data 
     */
    static void sgie_output_callback(GstBuffer *buf,
        NvDsInferNetworkInfo *network_info,
        NvDsInferLayerInfo *layers_info,
        guint num_layers,
        guint batch_size,
        gpointer user_data)
    {
        feature_callback_data_t* callback_data = reinterpret_cast<feature_callback_data_t*>(user_data);    

        /* Find the only output layer */
        NvDsInferLayerInfo *output_layer_info;
        NvDsInferLayerInfo *input_layer_info;
        for(int i = 0; i < num_layers; i++) {
            NvDsInferLayerInfo *info = &layers_info[i];
            if (info->isInput) {
                input_layer_info = info;
            } else {
                output_layer_info = info;
                // TODO: the info also include input tensor, which is the 3x112x112 input. COuld be use for something.
            }
        }

        /* Assign feature to NvDsFaceMetaData */
        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
        for(NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
            NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
            for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
                NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
                if (FACE_CLASS_ID != obj_meta->class_id)
                    continue;
                for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next) {
                    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
                    if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_OBJ_USER_META_FACE) {
                        NvDsFaceMetaData* faceMeta = reinterpret_cast<NvDsFaceMetaData*>(user_meta->user_meta_data);
                        // if (faceMeta->stage == NvDsFaceMetaStage::EMPTY)
                        //     continue;
                        // if (faceMeta->stage == NvDsFaceMetaStage::FEATURED)
                        //     continue;
                        // if (faceMeta->stage != NvDsFaceMetaStage::ALIGNED) {
                        //     printf(" %s:%d ERROR in feature, found an NVDS_OBJ_USER_META_FACE with stage = %d\n", __FILE__, __LINE__, faceMeta->stage);
                        //     exit(1);
                        // }
                        // faceMeta->stage = NvDsFaceMetaStage::FEATURED;
                        const int feature_size = output_layer_info->inferDims.numElements;
                        float* cur_feature = reinterpret_cast<float *>(output_layer_info->buffer) + 
                            faceMeta->aligned_index * feature_size;
                        memcpy(faceMeta->feature, cur_feature, feature_size * sizeof(float));

#ifdef DEBUG_FACE_SGIE
                        /* save input of sgie */
                        {
                            auto tensor_meta = faceMeta->aligned_tensor->tensor_meta;
                            int one_tensor_size = tensor_meta->buffer_size / tensor_meta->tensor_shape[0];
                            gpointer h_tensor = g_malloc(one_tensor_size);
                            g_assert(cudaSuccess == cudaMemcpy(h_tensor, 
                                tensor_meta->raw_tensor_buffer + faceMeta->aligned_index * one_tensor_size, 
                                one_tensor_size, cudaMemcpyDeviceToHost));

                            // frame-number_stream-number_object-number_object-type_widthxheight.jpg
                            const int MAX_STR_LEN = 1024;
                            char* filelocation = (char *)g_malloc0(MAX_STR_LEN);
                            snprintf(filelocation, MAX_STR_LEN - 1, "sgie_input_%d_%d_%d_roi_l%.f_t%.f_w%.f_h%.f.bin", 
                                frame_meta->frame_num,
                                frame_meta->source_id,
                                faceMeta->aligned_index,
                                obj_meta->detector_bbox_info.org_bbox_coords.left,
                                obj_meta->detector_bbox_info.org_bbox_coords.top,
                                obj_meta->detector_bbox_info.org_bbox_coords.width,
                                obj_meta->detector_bbox_info.org_bbox_coords.height
                                );

                            FILE *file = fopen (filelocation, "w");
                            if (!file) {
                                g_printerr ("Could not open file '%s' for writing:%s\n", filelocation, strerror (errno));
                            } else {
                                size_t written = fwrite (h_tensor, sizeof(float), one_tensor_size / sizeof(float), file);
                                if (written != one_tensor_size / sizeof(float)) {
                                    g_printerr("Error occurred at writing time to '%s'!\n", filelocation);
                                    exit(1);
                                }
                            }
                            fclose (file);

                            g_free(h_tensor);
                        }

                        /* save output of sgie */
                        {
                            const int MAX_STR_LEN = 1024;
                            char* filelocation = (char *)g_malloc0(MAX_STR_LEN);
                            snprintf(filelocation, MAX_STR_LEN - 1, "sgie_output_%d_%d_%d_roi_l%.f_t%.f_w%.f_h%.f.bin", 
                                frame_meta->frame_num,
                                frame_meta->source_id,
                                faceMeta->aligned_index,
                                obj_meta->detector_bbox_info.org_bbox_coords.left,
                                obj_meta->detector_bbox_info.org_bbox_coords.top,
                                obj_meta->detector_bbox_info.org_bbox_coords.width,
                                obj_meta->detector_bbox_info.org_bbox_coords.height
                                );

                            FILE *file = fopen (filelocation, "w");
                            if (!file) {
                                g_printerr ("Could not open file '%s' for writing:%s\n", filelocation, strerror (errno));
                            } else {
                                size_t written = fwrite (faceMeta->feature, sizeof(float), feature_size, file);
                                if (written != feature_size) {
                                    g_printerr("Error occurred at writing time to '%s'!\n", filelocation);
                                    exit(1);
                                }
                            }
                            fclose (file);
                        }
#endif // DEBUG_FACE_SGIE


                    }
                }
            }
        }
        callback_data->tensor_count++;
    }

FaceBin::FaceBin(FaceBinConfigs configs) : m_configs(configs)
{
    // createDetectBin();
    createFullBin();
}

void FaceBin::createDetectBin()
{
    m_masterBin = gst_bin_new("face-bin");
    GST_ASSERT(m_masterBin);

    m_backbone.pgie = gst_element_factory_make("nvinfer", "face-nvinfer");
    GST_ASSERT(m_backbone.pgie);

    GstPad *pgie_src_pad = gst_element_get_static_pad(m_backbone.pgie, "src");
    GST_ASSERT(pgie_src_pad);
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_face_src_pad_buffer_probe, nullptr, NULL);

    // Properties
    g_object_set(m_backbone.pgie, "config-file-path", m_configs.pgie_config_path, NULL);
    g_object_set(m_backbone.pgie, "output-tensor-meta", TRUE, NULL);
    g_object_set(m_backbone.pgie, "batch-size", 1, NULL);

    gst_bin_add_many(GST_BIN(m_masterBin), m_backbone.pgie, NULL);

    // Add ghost pads
    GstPad *pgie_sink_pad = gst_element_get_static_pad(m_backbone.pgie, "sink");
    GST_ASSERT(pgie_sink_pad);

    GstPad *sink_ghost_pad = gst_ghost_pad_new("sink", pgie_sink_pad);
    GST_ASSERT(sink_ghost_pad);

    GstPad *src_ghost_pad = gst_ghost_pad_new("src", pgie_src_pad);
    GST_ASSERT(src_ghost_pad);

    gst_pad_set_active(sink_ghost_pad, true);
    gst_pad_set_active(src_ghost_pad, true);

    gst_element_add_pad(m_masterBin, sink_ghost_pad);
    gst_element_add_pad(m_masterBin, src_ghost_pad);
    //
    gst_object_unref(pgie_src_pad);
}

void FaceBin::createFullBin()
{
    m_masterBin = gst_bin_new("face-bin");
    GST_ASSERT(m_masterBin);

    m_backbone.pgie = gst_element_factory_make("nvinfer", "face-nvinfer");
    GST_ASSERT(m_backbone.pgie);
    GstRegistry *registry;
    registry = gst_registry_get();
    gst_registry_scan_path(registry, "src/facealignment");

    GstPad *pgie_src_pad = gst_element_get_static_pad(m_backbone.pgie, "src");
    GST_ASSERT(pgie_src_pad);
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_face_src_pad_buffer_probe, nullptr, NULL);
    m_backbone.aligner = gst_element_factory_make("nvfacealign", "faceid-aligner");
    GST_ASSERT(m_backbone.aligner);

    m_backbone.sgie = gst_element_factory_make("nvinfer", "faceid-secondary-inference");
    GST_ASSERT(m_backbone.sgie);

    GstPad *sgie_src_pad = gst_element_get_static_pad(m_backbone.sgie, "src");
    GST_ASSERT(sgie_src_pad);
    // gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, sgie_face_src_pad_buffer_probe, nullptr, NULL);

    // Properties
    g_object_set(m_backbone.pgie, "config-file-path", m_configs.pgie_config_path, NULL);
    g_object_set(m_backbone.pgie, "output-tensor-meta", TRUE, NULL);
    g_object_set(m_backbone.pgie, "batch-size", 1, NULL);
    g_object_set(m_backbone.aligner, "config-file-path", m_configs.aligner_config_path, NULL);

    g_object_set(m_backbone.sgie, "config-file-path", m_configs.sgie_config_path, NULL);
    g_object_set(m_backbone.sgie, "input-tensor-meta", TRUE, NULL);
    g_object_set(m_backbone.sgie, "output-tensor-meta", TRUE, NULL);

    feature_callback_data_t *callback_data = new feature_callback_data_t;
    gst_nvinfer_raw_output_generated_callback out_callback = sgie_output_callback;
    g_object_set(m_backbone.sgie, "raw-output-generated-callback", out_callback, NULL);
    g_object_set(m_backbone.sgie, "raw-output-generated-userdata", reinterpret_cast<void *>(callback_data), NULL);

    gst_bin_add_many(GST_BIN(m_masterBin), m_backbone.pgie, m_backbone.aligner, m_backbone.sgie, NULL);
    gst_element_link_many(m_backbone.pgie, m_backbone.aligner, m_backbone.sgie, NULL);

    // Add ghost pads
    GstPad *pgie_sink_pad = gst_element_get_static_pad(m_backbone.pgie, "sink");
    GST_ASSERT(pgie_sink_pad);

    GstPad *sink_ghost_pad = gst_ghost_pad_new("sink", pgie_sink_pad);
    GST_ASSERT(sink_ghost_pad);

    // GstPad *src_ghost_pad = gst_ghost_pad_new("src", pgie_src_pad);
    GstPad *src_ghost_pad = gst_ghost_pad_new("src", sgie_src_pad);

    GST_ASSERT(src_ghost_pad);

    gst_pad_set_active(sink_ghost_pad, true);
    gst_pad_set_active(src_ghost_pad, true);

    gst_element_add_pad(m_masterBin, sink_ghost_pad);
    gst_element_add_pad(m_masterBin, src_ghost_pad);

    gst_object_unref(pgie_src_pad);
    gst_object_unref(pgie_sink_pad);
    gst_object_unref(sgie_src_pad);
}

void FaceBin::getMasterBin(GstElement *&bin)
{
    bin = this->m_masterBin;
}