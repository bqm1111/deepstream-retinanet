#include "face_bin.h"

FaceBin::FaceBin(FaceBinConfigs configs) : m_configs(configs)
{
    createBin();
}

void FaceBin::createBin()
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

    gst_object_unref(pgie_src_pad);
}

void FaceBin::getMasterBin(GstElement *&bin)
{
    bin = this->m_masterBin;
}