#include "detection_postprocess.h"

#include "nvdsinfer_custom_impl.h"
// #include "nvdsinfer_context.h"
#include "nvds_version.h"
#include "cuda_runtime.h"

#include <algorithm>

// #define MEASURE_POSTPROCESS_PERF_bfbe8afd24f9386fa6d18730bc496c65
#ifdef MEASURE_POSTPROCESS_PERF_bfbe8afd24f9386fa6d18730bc496c65
#include <chrono>
#endif
    /**
     * @brief implement NvDsMetaCopyFunc
     *
     * @param data
     * @param user_data
     * @return gpointer
     */
    gpointer nvds_copy_facemark_meta(gpointer data, gpointer user_data)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(data);
        NvDsFaceMetaData *facemark_meta_data_ptr = reinterpret_cast<faceid::NvDsFaceMetaData *>(
            user_meta->user_meta_data);
        NvDsFaceMetaData *new_facemark_meta_data_ptr = reinterpret_cast<faceid::NvDsFaceMetaData *>(
            g_memdup(facemark_meta_data_ptr, sizeof(faceid::NvDsFaceMetaData)));
        return reinterpret_cast<gpointer>(new_facemark_meta_data_ptr);
    }

    /**
     * @brief implement NvDsMetaReleaseFun
     *
     * @param data
     * @param user_data
     */
    void nvds_release_facemark_data(gpointer data, gpointer user_data)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(data);
        faceid::NvDsFaceMetaData *facemark_meta_data_ptr = reinterpret_cast<faceid::NvDsFaceMetaData *>(
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

    void nms(std::vector<Detection> &res, float *output, float post_cluster_thresh = 0.7, float iou_threshold = 0.4)
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
    
    GstPadProbeReturn pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
    {
#ifdef MEASURE_POSTPROCESS_PERF_bfbe8afd24f9386fa6d18730bc496c65
        auto begin = std::chrono::high_resolution_clock::now();
#endif
        GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
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
#ifdef MEASURE_POSTPROCESS_PERF_bfbe8afd24f9386fa6d18730bc496c65
                        auto _begin = std::chrono::high_resolution_clock::now();
#endif
                        cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                                   info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
#ifdef MEASURE_POSTPROCESS_PERF_bfbe8afd24f9386fa6d18730bc496c65
                        auto _end = std::chrono::high_resolution_clock::now();
                        double _duration = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _begin).count();
                        g_print(" %s:%d cudaMemcpy takes %f (ms)\n", __FILE__, __LINE__, _duration);
#endif
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
                    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)nvds_copy_facemark_meta;
                    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)nvds_release_facemark_data;
                    nvds_add_user_meta_to_obj(obj_meta, user_meta);
                    nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
                }
            }
        }

        if (nvds_enable_latency_measurement) {
            NvDsFrameLatencyInfo LATENCY_INFO[2];
            nvds_measure_buffer_latency(buf, LATENCY_INFO);
            g_print(" %s Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
                __func__,
                LATENCY_INFO[0].source_id,
                LATENCY_INFO[0].frame_num,
                LATENCY_INFO[0].latency
            );
        }
#ifdef MEASURE_POSTPROCESS_PERF_bfbe8afd24f9386fa6d18730bc496c65
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        g_print(" %s:%d %s takes %f (ms)\n", __FILE__, __LINE__, __func__, duration);
#endif
        return GST_PAD_PROBE_OK;
    }
