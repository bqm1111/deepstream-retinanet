#ifndef BUFFER_PROBE_H
#define BUFFER_PROBE_H
#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstmessage.h>
#include <gst/gstpipeline.h>
#include <gst/gstpoll.h>
#include <gst/gstvalue.h>
#include <iostream>
#include <nvdsmeta.h>
#include <gstnvdsinfer.h>
#include "gstnvdsmeta.h"
#include <nvdsinfer_custom_impl.h>
#include <nvdsmeta_schema.h>
#include "helper_function.h"
#include "params.h"
#include "cuda.h"
#include <algorithm>
#ifndef MAX_DISPLAY_LEN
#define MAX_DISPLAY_LEN 64
#endif
#define MAX_TIME_STAMP_LEN 32
#define PGIE_CLASS_ID_VEHICLE 2
#define PGIE_CLASS_ID_PERSON 0

GstPadProbeReturn osd_face_sink_pad_callback(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
GstPadProbeReturn pgie_face_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
GstPadProbeReturn tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);

GstPadProbeReturn osd_yolo_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);
GstPadProbeReturn pgie_yolo_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);


#endif