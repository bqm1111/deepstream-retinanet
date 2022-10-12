#include <json-glib/json-glib.h>
#include <uuid.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include "custom_schema.h"
#include "params.h"

gchar *generate_XFaceRawMeta_message(NvDsEventMsgMeta *meta)
{
	JsonNode *rootNode;
	JsonObject *rootObj;
	JsonObject *propObj;
	JsonObject *jObj;

	gchar *message;
	rootObj = json_object_new();
	propObj = json_object_new();

	// add frame info
	XFaceMetaMsg *msg_meta_content = (XFaceMetaMsg *)meta->extMsg;
	// json_object_set_string_member(rootObj, "timestamp", msg_meta_content->timestamp);
	json_object_set_string_member(rootObj, "title", g_strdup("RawMeta"));
	json_object_set_string_member(rootObj, "description", g_strdup("metadata of faces and person boxes found in each frame from video sources"));
	json_object_set_string_member(rootObj, "type", g_strdup("object"));

	// Required
	JsonArray *jRawMetaPropRequired = json_array_sized_new(6);
	json_array_add_string_element(jRawMetaPropRequired, g_strdup("timestamp"));
	json_array_add_string_element(jRawMetaPropRequired, g_strdup("camera_id"));
	json_array_add_string_element(jRawMetaPropRequired, g_strdup("frame_id"));
	json_array_add_string_element(jRawMetaPropRequired, g_strdup("session_id"));
	json_array_add_string_element(jRawMetaPropRequired, g_strdup("FACE"));
	json_array_add_string_element(jRawMetaPropRequired, g_strdup("MOT"));

	json_object_set_array_member(propObj, "required", jRawMetaPropRequired);

	// timestamp
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("Time stamp of this event message"));
	json_object_set_string_member(jObj, "type", g_strdup("double"));
	json_object_set_double_member(jObj, "value", msg_meta_content->timestamp);

	json_object_set_object_member(propObj, "timestamp", jObj);

	// Camera_id
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("camera_id of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("string"));
	json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->cameraId));
	json_object_set_object_member(propObj, "camera_id", jObj);

	// Frame_id
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("frame_id of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("integer"));
	json_object_set_int_member(jObj, "value", msg_meta_content->frameId);
	json_object_set_object_member(propObj, "frame_id", jObj);

	// session_id
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("session_id of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("string"));
	json_object_set_string_member(jObj, "value", msg_meta_content->sessionId);
	json_object_set_object_member(propObj, "session_id", jObj);

	// FACE
	JsonObject *faceArrObj = json_object_new();
	json_object_set_string_member(faceArrObj, "description", g_strdup("list of all faces in this frame"));
	json_object_set_string_member(faceArrObj, "type", g_strdup("array"));
	JsonArray *jFaceMetaArray = json_array_sized_new(msg_meta_content->num_face_obj);
	for (int i = 0; i < msg_meta_content->num_face_obj; i++)
	{
		JsonObject *faceObj = json_object_new();
		json_object_set_string_member(faceObj, "title", g_strdup("FaceRawMeta"));
		json_object_set_string_member(faceObj, "description", g_strdup("Face raw metadata"));
		json_object_set_string_member(faceObj, "type", g_strdup("object"));

		JsonObject *jbboxObj = json_object_new();
		json_object_set_string_member(jbboxObj, "title", g_strdup("Bbox"));
		json_object_set_string_member(jbboxObj, "description", g_strdup("Bounding box"));
		json_object_set_string_member(jbboxObj, "type", g_strdup("object"));
		// x
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "top left x coordinate of face image");
		json_object_set_string_member(jObj, "type", "number");
		json_object_set_double_member(jObj, "value", msg_meta_content->face_meta_list[i]->bbox.top);
		json_object_set_object_member(jbboxObj, "x", jObj);
		// y
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "top left y coordinate of face image");
		json_object_set_string_member(jObj, "type", "number");
		json_object_set_double_member(jObj, "value", msg_meta_content->face_meta_list[i]->bbox.left);
		json_object_set_object_member(jbboxObj, "y", jObj);
		// w
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "width of face image");
		json_object_set_string_member(jObj, "type", "number");
		json_object_set_double_member(jObj, "value", msg_meta_content->face_meta_list[i]->bbox.width);
		json_object_set_object_member(jbboxObj, "w", jObj);
		// h
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "height of face image");
		json_object_set_string_member(jObj, "type", "number");
		json_object_set_double_member(jObj, "value", msg_meta_content->face_meta_list[i]->bbox.height);
		json_object_set_object_member(jbboxObj, "h", jObj);

		JsonArray *jbboxRequired = json_array_sized_new(4);
		json_array_add_string_element(jbboxRequired, g_strdup("x"));
		json_array_add_string_element(jbboxRequired, g_strdup("y"));
		json_array_add_string_element(jbboxRequired, g_strdup("w"));
		json_array_add_string_element(jbboxRequired, g_strdup("h"));

		json_object_set_array_member(jbboxObj, "required", jbboxRequired);
		json_object_set_object_member(faceObj, "bbox", jbboxObj);

		// confidence_score
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "confidence score of name of the person appeared on the face image");
		json_object_set_string_member(jObj, "type", "float");
		json_object_set_double_member(jObj, "value", msg_meta_content->face_meta_list[i]->confidence_score);
		json_object_set_object_member(faceObj, "confidence_score", jObj);

		// name
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "name of the person appeared on the face image");
		json_object_set_string_member(jObj, "type", "string");
		json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->face_meta_list[i]->name));
		json_object_set_object_member(faceObj, "name", jObj);
		// staff_id
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "staff_id of the person appeared on the face image");
		json_object_set_string_member(jObj, "type", "string");
		json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->face_meta_list[i]->staff_id));
		json_object_set_object_member(faceObj, "staff_id", jObj);

		// feature
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "vector feature of face image");
		json_object_set_string_member(jObj, "type", "bytes");
		json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->face_meta_list[i]->feature));
		json_object_set_object_member(faceObj, "feature", jObj);

		// encoded_img
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "jpeg encoded image of face");
		json_object_set_string_member(jObj, "type", "bytes");
		json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->face_meta_list[i]->encoded_img));
		json_object_set_object_member(faceObj, "encoded_img", jObj);

		JsonArray *jFaceRequired = json_array_sized_new(6);
		json_array_add_string_element(jFaceRequired, g_strdup("bbox"));
		json_array_add_string_element(jFaceRequired, g_strdup("confidence_score"));
		json_array_add_string_element(jFaceRequired, g_strdup("name"));
		json_array_add_string_element(jFaceRequired, g_strdup("staff_id"));
		json_array_add_string_element(jFaceRequired, g_strdup("feature"));
		json_array_add_string_element(jFaceRequired, g_strdup("encoded_img"));
		json_object_set_array_member(faceObj, "required", jFaceRequired);

		json_array_add_object_element(jFaceMetaArray, faceObj);
	}

	json_object_set_array_member(propObj, "FACE", jFaceMetaArray);

	// MOT
	JsonObject *motArrObj = json_object_new();
	json_object_set_string_member(motArrObj, "description", g_strdup("list of all person boxes in this frame"));
	json_object_set_string_member(motArrObj, "type", g_strdup("array"));
	JsonArray *jMOTMetaArray = json_array_sized_new(msg_meta_content->num_mot_obj);
	for (int i = 0; i < msg_meta_content->num_mot_obj; i++)
	{
		JsonObject *motObj = json_object_new();
		json_object_set_string_member(motObj, "title", g_strdup("MOTRawMeta"));
		json_object_set_string_member(motObj, "description", g_strdup("MOT raw metadata"));
		json_object_set_string_member(motObj, "type", g_strdup("object"));

		JsonObject *jbboxObj = json_object_new();
		json_object_set_string_member(jbboxObj, "title", g_strdup("Bbox"));
		json_object_set_string_member(jbboxObj, "description", g_strdup("Bounding box"));
		json_object_set_string_member(jbboxObj, "type", g_strdup("object"));
		// x
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "top left x coordinate of personBox image");
		json_object_set_string_member(jObj, "type", "number");
		json_object_set_double_member(jObj, "value", msg_meta_content->mot_meta_list[i]->bbox.top);
		json_object_set_object_member(jbboxObj, "x", jObj);
		// y
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "top left y coordinate of personBox image");
		json_object_set_string_member(jObj, "type", "number");
		json_object_set_double_member(jObj, "value", msg_meta_content->mot_meta_list[i]->bbox.left);
		json_object_set_object_member(jbboxObj, "y", jObj);
		// w
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "width of personBox image");
		json_object_set_string_member(jObj, "type", "number");
		json_object_set_double_member(jObj, "value", msg_meta_content->mot_meta_list[i]->bbox.width);
		json_object_set_object_member(jbboxObj, "w", jObj);
		// h
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "height of personBox image");
		json_object_set_string_member(jObj, "type", "number");
		json_object_set_double_member(jObj, "value", msg_meta_content->mot_meta_list[i]->bbox.height);
		json_object_set_object_member(jbboxObj, "h", jObj);

		JsonArray *jbboxRequired = json_array_sized_new(4);
		json_array_add_string_element(jbboxRequired, g_strdup("x"));
		json_array_add_string_element(jbboxRequired, g_strdup("y"));
		json_array_add_string_element(jbboxRequired, g_strdup("w"));
		json_array_add_string_element(jbboxRequired, g_strdup("h"));

		json_object_set_array_member(jbboxObj, "required", jbboxRequired);
		json_object_set_object_member(motObj, "bbox", jbboxObj);

		// track_id
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "track_id of the person appeared on the face image");
		json_object_set_string_member(jObj, "type", "integer");
		json_object_set_int_member(jObj, "value", msg_meta_content->mot_meta_list[i]->track_id);
		json_object_set_object_member(motObj, "track_id", jObj);

		// feature
		jObj = json_object_new();
		json_object_set_string_member(jObj, "description", "vector embedding of this personBox");
		json_object_set_string_member(jObj, "type", "bytes");
		json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->mot_meta_list[i]->embedding));
		json_object_set_object_member(motObj, "embedding", jObj);

		JsonArray *jMOTRequired = json_array_sized_new(3);
		json_array_add_string_element(jMOTRequired, g_strdup("bbox"));
		json_array_add_string_element(jMOTRequired, g_strdup("track_id"));
		json_array_add_string_element(jMOTRequired, g_strdup("embedding"));
		json_object_set_array_member(motObj, "required", jMOTRequired);

		json_array_add_object_element(jMOTMetaArray, motObj);
	}

	json_object_set_array_member(propObj, "MOT", jMOTMetaArray);

	json_object_set_object_member(rootObj, "properties", propObj);
	// create root node
	rootNode = json_node_new(JSON_NODE_OBJECT);
	json_node_set_object(rootNode, rootObj);

	// create message
	message = json_to_string(rootNode, TRUE);

	json_node_free(rootNode);
	json_object_unref(rootObj);

	return message;
}

gchar *generate_XFace_visual_message(NvDsEventMsgMeta *meta)
{
	JsonNode *rootNode;
	JsonObject *rootObj;
	JsonObject *propObj;
	JsonObject *jObj;

	gchar *message;
	rootObj = json_object_new();
	propObj = json_object_new();

	// add frame info
	XFaceVisualMsg *msg_meta_content = (XFaceVisualMsg *)meta->extMsg;
	// json_object_set_string_member(rootObj, "timestamp", msg_meta_content->timestamp);
	json_object_set_string_member(rootObj, "title", g_strdup("HDImage"));
	json_object_set_string_member(rootObj, "description", g_strdup("HDImage of each frame from video sources"));
	json_object_set_string_member(rootObj, "type", g_strdup("object"));

	// Required
	JsonArray *jVisualPropRequired = json_array_sized_new(8);
	json_array_add_string_element(jVisualPropRequired, g_strdup("timestamp"));
	json_array_add_string_element(jVisualPropRequired, g_strdup("camera_id"));
	json_array_add_string_element(jVisualPropRequired, g_strdup("frame_id"));
	json_array_add_string_element(jVisualPropRequired, g_strdup("session_id"));
	json_array_add_string_element(jVisualPropRequired, g_strdup("width"));
	json_array_add_string_element(jVisualPropRequired, g_strdup("height"));
	json_array_add_string_element(jVisualPropRequired, g_strdup("channel"));
	json_array_add_string_element(jVisualPropRequired, g_strdup("image"));

	json_object_set_array_member(propObj, "required", jVisualPropRequired);

	// timestamp
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("Time stamp of this event message"));
	json_object_set_string_member(jObj, "type", g_strdup("double"));
	json_object_set_double_member(jObj, "value", msg_meta_content->timestamp);

	json_object_set_object_member(propObj, "timestamp", jObj);

	// Camera_id
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("camera_id of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("string"));
	json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->cameraId));
	json_object_set_object_member(propObj, "camera_id", jObj);

	// Frame_id
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("frame_id of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("integer"));
	json_object_set_int_member(jObj, "value", msg_meta_content->frameId);
	json_object_set_object_member(propObj, "frame_id", jObj);

	// session_id
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("session_id of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("string"));
	json_object_set_string_member(jObj, "value", msg_meta_content->sessionId);
	json_object_set_object_member(propObj, "frame_id", jObj);
	// width
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("witdh of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("integer"));
	json_object_set_int_member(jObj, "value", msg_meta_content->width);
	json_object_set_object_member(propObj, "width", jObj);
	// height
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("height of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("integer"));
	json_object_set_int_member(jObj, "value", msg_meta_content->height);
	json_object_set_object_member(propObj, "height", jObj);
	// num_channel
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("number of channel of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("integer"));
	json_object_set_int_member(jObj, "value", msg_meta_content->num_channel);
	json_object_set_object_member(propObj, "channel", jObj);
	// bas264 encoded image
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", g_strdup("bas264 encoded image of this frame"));
	json_object_set_string_member(jObj, "type", g_strdup("bytes"));
	json_object_set_string_member(jObj, "value", msg_meta_content->full_img);
	json_object_set_object_member(propObj, "image", jObj);

	json_object_set_object_member(rootObj, "properties", propObj);
	// create root node
	rootNode = json_node_new(JSON_NODE_OBJECT);
	json_node_set_object(rootNode, rootObj);

	// create message
	message = json_to_string(rootNode, TRUE);

	json_node_free(rootNode);
	json_object_unref(rootObj);

	return message;
}

gchar *
generate_XFace_event_message(void *privData, NvDsEventMsgMeta *meta)
{
	gchar *message;
	switch (meta->componentId)
	{
	case 1:
		message = generate_XFaceRawMeta_message(meta);
		break;
	case 2:
		message = generate_XFace_visual_message(meta);
		break;
	default:
		break;
	}
	return message;
}
