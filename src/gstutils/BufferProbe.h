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
#ifndef MAX_DISPLAY_LEN
#define MAX_DISPLAY_LEN 64
#endif

static GstPadProbeReturn osd_sink_pad_callback(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
#endif