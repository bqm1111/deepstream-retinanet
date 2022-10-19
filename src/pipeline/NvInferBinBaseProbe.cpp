#include "NvInferBinBase.h"
#include <chrono>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include "message.h"
GstPadProbeReturn NvInferBinBase::osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GST_ASSERT(buf);
    if (!buf)
    {
        return GST_PAD_PROBE_OK;
    }

    if (_udata != nullptr)
    {
        /* do speed mesurement */
        SinkPerfStruct *sink_perf_struct = reinterpret_cast<SinkPerfStruct *>(_udata);
        sink_perf_struct->check_start();
        sink_perf_struct->update();
        sink_perf_struct->log();

        if (nvds_enable_latency_measurement)
        {
            NvDsFrameLatencyInfo *latency_info = (NvDsFrameLatencyInfo *)malloc(20 * sizeof(NvDsFrameLatencyInfo));
            int num_sources_in_batch = nvds_measure_buffer_latency(buf, latency_info);
            for (int i = 0; i < num_sources_in_batch; i++)
            {
                QDTLOG_DEBUG("source_id={} frame_num={} frame latancy={}",
                              latency_info[i].source_id,
                              latency_info[i].frame_num,
                              latency_info[i].latency);
            }
            free(latency_info);
        }
    }

    return GST_PAD_PROBE_OK;
}


GstPadProbeReturn NvInferBinBase::tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    return GST_PAD_PROBE_OK;
}
