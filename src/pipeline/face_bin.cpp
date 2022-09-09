#include "face_bin.h"

FaceBin::FaceBin(FaceBinConfigs configs) : m_configs(configs)
{
    createDetectBin();
    // createFullBin();
}

void FaceBin::createDetectBin()
{
    m_masterBin = gst_bin_new("face-bin");
    GST_ASSERT(m_masterBin);

    m_backbone.pgie = gst_element_factory_make("nvinfer", "face-nvinfer");
    GST_ASSERT(m_backbone.pgie);

    GstPad *pgie_src_pad = gst_element_get_static_pad(m_backbone.pgie, "src");
    GST_ASSERT(pgie_src_pad);
    // gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_face_src_pad_buffer_probe, nullptr, NULL);

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
    gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, sgie_face_src_pad_buffer_probe, nullptr, NULL);

    // Properties
    g_object_set(m_backbone.pgie, "config-file-path", m_configs.pgie_config_path, NULL);
    g_object_set(m_backbone.pgie, "output-tensor-meta", TRUE, NULL);
    g_object_set(m_backbone.pgie, "batch-size", 1, NULL);
    g_object_set(m_backbone.aligner, "config-file-path", m_configs.aligner_config_path, NULL);

    g_object_set(m_backbone.sgie, "config-file-path", m_configs.sgie_config_path, NULL);
    g_object_set(m_backbone.sgie, "input-tensor-meta", TRUE, NULL);
    g_object_set(m_backbone.sgie, "output-tensor-meta", TRUE, NULL);

    // feature_callback_data_t *callback_data = new feature_callback_data_t;
    // gst_nvinfer_raw_output_generated_callback out_callback = write_feature_output_to_file;
    // g_object_set(m_backbone.sgie, "raw-output-generated-callback", out_callback, NULL);
    // g_object_set(m_backbone.sgie, "raw-output-generated-userdata", reinterpret_cast<void *>(callback_data), NULL);

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