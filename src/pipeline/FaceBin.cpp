#include "FaceBin.h"
#include "QDTLog.h"
void NvInferFaceBin::setConfig(std::shared_ptr<NvInferFaceBinConfig> configs)
{
    m_configs = configs;
    m_module_name = "face";
}

void NvInferFaceBin::createInferBin()
{
    m_masterBin = gst_bin_new("face-bin");
    GST_ASSERT(m_masterBin);

    m_pgie = gst_element_factory_make("nvinfer", "face-nvinfer");
    GST_ASSERT(m_pgie);
    GstRegistry *registry;
    registry = gst_registry_get();
    gst_registry_scan_path(registry, "src/facealignment");

    GstPad *pgie_src_pad = gst_element_get_static_pad(m_pgie, "src");
    GST_ASSERT(pgie_src_pad);
    m_obj_ctx_handle = nvds_obj_enc_create_context();
    if (!m_obj_ctx_handle)
    {
        QDTLog::error("%s:%d Unable to create context\n", __FILE__, __LINE__);
    }
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, this->pgie_src_pad_buffer_probe,
                      (gpointer)m_obj_ctx_handle, NULL);
    m_aligner = gst_element_factory_make("nvfacealign", "faceid-aligner");
    GST_ASSERT(m_aligner);

    m_sgie = gst_element_factory_make("nvinfer", "faceid-secondary-inference");
    GST_ASSERT(m_sgie);

    GstPad *sgie_src_pad = gst_element_get_static_pad(m_sgie, "src");
    GST_ASSERT(sgie_src_pad);
    // Properties
    g_object_set(m_pgie, "config-file-path", m_configs->pgie_config_path.c_str(), NULL);
    g_object_set(m_pgie, "output-tensor-meta", TRUE, NULL);
    g_object_set(m_aligner, "config-file-path", std::dynamic_pointer_cast<NvInferFaceBinConfig>(m_configs)->aligner_config_path.c_str(), NULL);

    g_object_set(m_sgie, "config-file-path", m_configs->sgie_config_path.c_str(), NULL);
    g_object_set(m_sgie, "input-tensor-meta", TRUE, NULL);
    g_object_set(m_sgie, "output-tensor-meta", TRUE, NULL);

    user_callback_data *callback_data = m_user_callback_data;
    gst_nvinfer_raw_output_generated_callback out_callback = this->sgie_output_callback;
    g_object_set(m_sgie, "raw-output-generated-callback", out_callback, NULL);
    g_object_set(m_sgie, "raw-output-generated-userdata", reinterpret_cast<void *>(callback_data), NULL);

    gst_bin_add_many(GST_BIN(m_masterBin), m_pgie, m_aligner, m_sgie, NULL);
    gst_element_link_many(m_pgie, m_aligner, m_sgie, NULL);

    // Add ghost pads
    GstPad *pgie_sink_pad = gst_element_get_static_pad(m_pgie, "sink");
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

    m_pgie = gst_element_factory_make("nvinfer", "face-nvinfer");
    GST_ASSERT(m_pgie);

    GstPad *pgie_src_pad = gst_element_get_static_pad(m_pgie, "src");
    GST_ASSERT(pgie_src_pad);
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, this->pgie_src_pad_buffer_probe, nullptr, NULL);

    // Properties
    g_object_set(m_pgie, "config-file-path", m_configs->pgie_config_path.c_str(), NULL);
    g_object_set(m_pgie, "output-tensor-meta", TRUE, NULL);

    gst_bin_add_many(GST_BIN(m_masterBin), m_pgie, NULL);

    // Add ghost pads
    GstPad *pgie_sink_pad = gst_element_get_static_pad(m_pgie, "sink");
    GST_ASSERT(pgie_sink_pad);

    GstPad *sink_ghost_pad = gst_ghost_pad_new("sink", pgie_sink_pad);
    GST_ASSERT(sink_ghost_pad);

    GstPad *src_ghost_pad = gst_ghost_pad_new("src", pgie_src_pad);
    GST_ASSERT(src_ghost_pad);

    gst_pad_set_active(sink_ghost_pad, true);
    gst_pad_set_active(src_ghost_pad, true);

    gst_element_add_pad(m_masterBin, sink_ghost_pad);
    gst_element_add_pad(m_masterBin, src_ghost_pad);

    gst_object_unref(pgie_src_pad);
}

void NvInferFaceBin::attachProbe()
{
    GstPad *tiler_sink_pad = gst_element_get_static_pad(m_tiler, "sink");
    GST_ASSERT(tiler_sink_pad);
    gst_pad_add_probe(tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, tiler_sink_pad_buffer_probe,
                      reinterpret_cast<gpointer>(m_tiler), NULL);
    g_object_unref(tiler_sink_pad);

    GstPad *osd_sink_pad = gst_element_get_static_pad(m_osd, "sink");
    GST_ASSERT(osd_sink_pad);
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe,
                      reinterpret_cast<gpointer>(m_tiler), NULL);
    gst_object_unref(osd_sink_pad);
}
