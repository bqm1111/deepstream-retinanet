#include "FaceBin.h"

void NvInferFaceBin::createBin()
{
    m_masterBin = gst_bin_new("face-bin");
    GST_ASSERT(m_masterBin);

    pgie = gst_element_factory_make("nvinfer", "face-nvinfer");
    GST_ASSERT(pgie);
    GstRegistry *registry;
    registry = gst_registry_get();
    gst_registry_scan_path(registry, "src/facealignment");

    GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
    GST_ASSERT(pgie_src_pad);
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, this->pgie_src_pad_buffer_probe, nullptr, NULL);
    aligner = gst_element_factory_make("nvfacealign", "faceid-aligner");
    GST_ASSERT(aligner);

    sgie = gst_element_factory_make("nvinfer", "faceid-secondary-inference");
    GST_ASSERT(sgie);

    GstPad *sgie_src_pad = gst_element_get_static_pad(sgie, "src");
    GST_ASSERT(sgie_src_pad);
    // gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, sgie_face_src_pad_buffer_probe, nullptr, NULL);
    // Properties
    g_object_set(pgie, "config-file-path", m_configs->pgie_config_path.c_str(), NULL);
    g_object_set(pgie, "output-tensor-meta", TRUE, NULL);
    g_object_set(pgie, "batch-size", 1, NULL);
    g_object_set(aligner, "config-file-path", std::dynamic_pointer_cast<NvInferFaceBinConfig>(m_configs)->aligner_config_path.c_str(), NULL);

    g_object_set(sgie, "config-file-path", m_configs->sgie_config_path.c_str(), NULL);
    g_object_set(sgie, "input-tensor-meta", TRUE, NULL);
    g_object_set(sgie, "output-tensor-meta", TRUE, NULL);

    user_feature_callback_data_t *callback_data = new user_feature_callback_data_t;
    callback_data->curl = m_curl;
    gst_nvinfer_raw_output_generated_callback out_callback = this->sgie_output_callback;
    g_object_set(sgie, "raw-output-generated-callback", out_callback, NULL);
    g_object_set(sgie, "raw-output-generated-userdata", reinterpret_cast<void *>(callback_data), NULL);

    gst_bin_add_many(GST_BIN(m_masterBin), pgie, aligner, sgie, NULL);
    gst_element_link_many(pgie, aligner, sgie, NULL);

    // Add ghost pads
    GstPad *pgie_sink_pad = gst_element_get_static_pad(pgie, "sink");
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

void NvInferFaceBin::createDetectBin()
{
    m_masterBin = gst_bin_new("face-bin");
    GST_ASSERT(m_masterBin);

    pgie = gst_element_factory_make("nvinfer", "face-nvinfer");
    GST_ASSERT(pgie);

    GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
    GST_ASSERT(pgie_src_pad);
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, this->pgie_src_pad_buffer_probe, nullptr, NULL);

    // Properties
    g_object_set(pgie, "config-file-path", m_configs->pgie_config_path.c_str(), NULL);
    g_object_set(pgie, "output-tensor-meta", TRUE, NULL);
    g_object_set(pgie, "batch-size", 1, NULL);

    gst_bin_add_many(GST_BIN(m_masterBin), pgie, NULL);

    // Add ghost pads
    GstPad *pgie_sink_pad = gst_element_get_static_pad(pgie, "sink");
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

void NvInferFaceBin::acquireCurl(CURL * curl)
{
    m_curl = curl;
}
