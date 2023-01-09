#include "MOTBin.h"
#include <memory>

void NvInferMOTBin::setConfig(std::shared_ptr<NvInferMOTBinConfig> configs)
{
    m_configs = configs;
    m_module_name = "mot";
}

void NvInferMOTBin::createInferBin()
{
    m_masterBin = gst_bin_new("mot-bin");
    GST_ASSERT(m_masterBin);

    m_pgie = gst_element_factory_make("nvinfer", "mot-primary-nvinfer");
    GST_ASSERT(m_pgie);

    GstPad *pgie_src_pad = gst_element_get_static_pad(m_pgie, "src");
    GST_ASSERT(pgie_src_pad);

    m_sgie = gst_element_factory_make("nvinfer", "mot-secondary-inference");
    GST_ASSERT(m_sgie);

    GstPad *sgie_src_pad = gst_element_get_static_pad(m_sgie, "src");
    GST_ASSERT(sgie_src_pad);

    // Properties
    g_object_set(m_pgie, "config-file-path", m_configs->pgie_config_path.c_str(), NULL);

    g_object_set(m_sgie, "config-file-path", m_configs->sgie_config_path.c_str(), NULL);

    gst_bin_add_many(GST_BIN(m_masterBin), m_pgie, m_sgie, NULL);
    gst_element_link_many(m_pgie, m_sgie, NULL);

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

    // add probes
    GstPad *sgie_srcpad = gst_element_get_static_pad(m_sgie, "src");
    if (!sgie_srcpad)
    {
        gst_print("no pad with name \"src\" found for secondary-inference\n");
        gst_object_unref(sgie_srcpad);
        throw std::runtime_error("");
    }
    gst_pad_add_probe(sgie_srcpad, GST_PAD_PROBE_TYPE_BUFFER,
                      sgie_src_pad_buffer_probe, m_user_callback_data, NULL);
    gst_object_unref(sgie_srcpad);
}

void NvInferMOTBin::attachProbe()
{
    // GstPad *tiler_sink_pad = gst_element_get_static_pad(m_tiler, "sink");
    // GST_ASSERT(tiler_sink_pad);
    // gst_pad_add_probe(tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, tiler_sink_pad_buffer_probe,
    //                   reinterpret_cast<gpointer>(m_tiler), NULL);
    // g_object_unref(tiler_sink_pad);

    GstPad *osd_sink_pad = gst_element_get_static_pad(m_osd, "sink");
    GST_ASSERT(osd_sink_pad);
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe,
                      m_user_callback_data, NULL);
    gst_object_unref(osd_sink_pad);
}
