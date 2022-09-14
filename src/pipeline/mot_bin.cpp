#include "mot_bin.h"

#include "face_bin.h"

MOTBin::MOTBin(MOTBinConfigs configs) : m_configs(configs)
{
    createBin();
}

void MOTBin::createBin()
{
    m_masterBin = gst_bin_new("face-bin");
    GST_ASSERT(m_masterBin);

    m_backbone.pgie = gst_element_factory_make("nvinfer", "mot-primary-nvinfer");
    GST_ASSERT(m_backbone.pgie);

    GstPad *pgie_src_pad = gst_element_get_static_pad(m_backbone.pgie, "src");
    GST_ASSERT(pgie_src_pad);
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_face_src_pad_buffer_probe, nullptr, NULL);

    m_backbone.sgie = gst_element_factory_make("nvinfer", "mot-secondary-inference");
    GST_ASSERT(m_backbone.sgie);

    GstPad *sgie_src_pad = gst_element_get_static_pad(m_backbone.sgie, "src");
    GST_ASSERT(sgie_src_pad);
    // gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, sgie_face_src_pad_buffer_probe, nullptr, NULL);

    // Properties
    g_object_set(m_backbone.pgie, "config-file-path", m_configs.pgie_config_path, NULL);
    // g_object_set(m_backbone.pgie, "output-tensor-meta", TRUE, NULL);
    // g_object_set(m_backbone.pgie, "batch-size", 1, NULL);

    g_object_set(m_backbone.sgie, "config-file-path", m_configs.sgie_config_path, NULL);
    // g_object_set(m_backbone.sgie, "input-tensor-meta", TRUE, NULL);
    // g_object_set(m_backbone.sgie, "output-tensor-meta", TRUE, NULL);

    gst_bin_add_many(GST_BIN(m_masterBin), m_backbone.pgie, m_backbone.sgie, NULL);
    gst_element_link_many(m_backbone.pgie, m_backbone.sgie, NULL);

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

    this->m_tracker_list = (MOTTrackerList *)g_malloc0(sizeof(MOTTrackerList));
    std::cout << "Num tracker = " << this->m_configs.num_trackers << std::endl;
    this->m_tracker_list->trackers = (tracker *)g_malloc0(sizeof(tracker) * this->m_configs.num_trackers);
    this->m_tracker_list->num_trackers = this->m_configs.num_trackers;
    for (size_t i = 0; i < this->m_tracker_list->num_trackers; i++)
        this->m_tracker_list->trackers[i] = tracker(
            0.1363697015033318, 91, 0.7510890862625559, 18, 2, 1.);

        // add probes
    GstPad *sgie_srcpad = gst_element_get_static_pad(this->m_backbone.sgie, "src");
    if (!sgie_srcpad) {
        gst_print ("no pad with name \"src\" found for secondary-inference\n");
        gst_object_unref(sgie_srcpad);
        throw std::runtime_error("");
    }
    gst_pad_add_probe(sgie_srcpad, GST_PAD_PROBE_TYPE_BUFFER, 
                      sgie_mot_src_pad_buffer_probe, this->m_tracker_list, NULL);
    gst_object_unref(sgie_srcpad);
}

void MOTBin::getMasterBin(GstElement *&bin)
{
    bin = this->m_masterBin;
}