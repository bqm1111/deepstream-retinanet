#ifndef _DETECTION_POSTPROCESS_H_
#define _DETECTION_POSTPROCESS_H_

#include <gst/gst.h>
#include "common.h"

namespace faceid
{
    GstPadProbeReturn pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
}


#endif