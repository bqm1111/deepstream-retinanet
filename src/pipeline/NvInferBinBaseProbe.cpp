#include "NvInferBinBase.h"
#include <chrono>

GstPadProbeReturn NvInferBinBase::osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    return GST_PAD_PROBE_OK;
}
// 
GstPadProbeReturn NvInferBinBase::tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
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
        if (!sink_perf_struct->start_perf_measurement)
        {
            sink_perf_struct->start_perf_measurement = true;
            sink_perf_struct->last_tick = std::chrono::high_resolution_clock::now();
        }
        else
        {
            // Measure
            auto tick = std::chrono::high_resolution_clock::now();
            double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(tick - sink_perf_struct->last_tick).count();

            // Update
            sink_perf_struct->last_tick = tick;
            sink_perf_struct->total_time += elapsed_time;
            sink_perf_struct->num_ticks++;

            // Statistics
            double avg_runtime = sink_perf_struct->total_time / sink_perf_struct->num_ticks / 1e6;
            double avg_fps = 1.0 / avg_runtime;
            g_print(" %s:%d Tiler Average runtime: %f  Average FPS: %f \n", __FILE__, __LINE__, avg_runtime, avg_fps);
        }

        if (nvds_enable_latency_measurement)
        {
            NvDsFrameLatencyInfo latency_info[2];
            nvds_measure_buffer_latency(buf, latency_info);
            g_print(" %s Source id = %d Frame_num = %d Frame latency = %lf (ms) \n",
                    __func__,
                    latency_info[0].source_id,
                    latency_info[0].frame_num,
                    latency_info[0].latency);
        }
    }

    return GST_PAD_PROBE_OK;
}
