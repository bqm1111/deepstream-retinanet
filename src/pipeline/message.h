#ifndef MESSAGE_H
#define MESSAGE_H
#include <json-glib/json-glib.h>
#include <nvdsmeta_schema.h>
#include "params.h"

gchar *generate_MOTRawMeta_message(XFaceMOTMsgMeta *msg_meta_content);
gchar *generate_FaceRawMeta_message(std::shared_ptr<NvDsFaceMsgData> msg_meta_content);
gchar *generate_XFace_visual_message(XFaceVisualMsg *msg_meta_content);

#endif