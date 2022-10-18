#ifndef MESSAGE_H
#define MESSAGE_H
#include <json-glib/json-glib.h>
#include <nvdsmeta_schema.h>
#include "params.h"

gchar *generate_XFaceRawMeta_message(NvDsEventMsgMeta *meta);
gchar *generate_XFace_visual_message(NvDsEventMsgMeta *meta);

#endif