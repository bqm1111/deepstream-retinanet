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
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data);
#endif