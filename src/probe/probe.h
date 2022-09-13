#ifndef BUFFER_PROBE_H
#define BUFFER_PROBE_H
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstmessage.h>
#include <gst/gstpipeline.h>
#include <gst/gstpoll.h>
#include <gst/gstvalue.h>
#include <nvdsmeta.h>
#include <nvdsinfer_custom_impl.h>
#include <nvdsmeta_schema.h>
#include "common.h"
#include "params.h"
#include "cuda.h"
#include <algorithm>
#include "track.h"
#include "tracker.h"
#include "mot_bin.h"

#ifndef MAX_DISPLAY_LEN
#define MAX_DISPLAY_LEN 64
#endif
#define MAX_TIME_STAMP_LEN 32
#define PGIE_CLASS_ID_VEHICLE 2
#define PGIE_CLASS_ID_PERSON 0

#define EMBEDDING_DIMS 512
#define POST_TRACK_SCORE 1.0

// FACE PROBE
struct feature_callback_data_t
{
    int tensor_count = 0;
};
void sgie_output_callback(GstBuffer *buf,
                          NvDsInferNetworkInfo *network_info,
                          NvDsInferLayerInfo *layers_info,
                          guint num_layers,
                          guint batch_size,
                          gpointer user_data);
GstPadProbeReturn osd_face_sink_pad_callback(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
GstPadProbeReturn pgie_face_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
GstPadProbeReturn sgie_face_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
GstPadProbeReturn tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);

// MOT PROBE
GstPadProbeReturn sgie_mot_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

// YOLO PROBE
GstPadProbeReturn osd_yolo_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);
GstPadProbeReturn pgie_yolo_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);

#endif