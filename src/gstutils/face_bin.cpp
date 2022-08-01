#include "face_bin.h"
#include "common.h"
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstpad.h>
#include "BufferProbe.h"

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
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_src_pad_buffer_probe, nullptr, NULL);
    gst_object_unref(pgie_src_pad);
}