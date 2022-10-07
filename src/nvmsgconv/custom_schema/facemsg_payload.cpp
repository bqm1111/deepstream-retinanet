#include <json-glib/json-glib.h>
#include <uuid.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include "custom_schema.h"
#include "params.h"

static JsonObject *
generate_object_object(void *privData, NvDsEventMsgMeta *meta)
{
	JsonObject *objectObj;
	JsonObject *jobject;
	guint i;
	gchar tracking_id[64];

	// object object
	objectObj = json_object_new();
	if (snprintf(tracking_id, sizeof(tracking_id), "%lu", meta->trackingId) >= (int)sizeof(tracking_id))
		g_warning("Not enough space to copy trackingId");
	// json_object_set_string_member(objectObj, "id", tracking_id);

	switch (meta->objType)
	{
	}

	// bbox sub object
	jobject = json_object_new();
	json_object_set_int_member(jobject, "topleftx", meta->bbox.left);
	json_object_set_int_member(jobject, "toplefty", meta->bbox.top);
	json_object_set_int_member(jobject, "bottomrightx", meta->bbox.left + meta->bbox.width);
	json_object_set_int_member(jobject, "bottomrighty", meta->bbox.top + meta->bbox.height);
	json_object_set_object_member(objectObj, "bbox", jobject);

	jobject = json_object_new();
	if (meta->extMsgSize > 0)
	{
		FaceEventMsgData *dsObj = (FaceEventMsgData *)meta->extMsg;
		if (dsObj)
		{
			json_object_set_string_member(jobject, "feature", (gchar *)dsObj->feature);
		}
	}
	json_object_set_object_member(objectObj, "feature", jobject);
	return objectObj;
}

gchar *generate_face_event_message(void *privData, NvDsEventMsgMeta *meta)
{
	JsonNode *rootNode;
	JsonObject *rootObj;
	JsonObject *objectObj;
	gchar *message;

	uuid_t msgId;
	gchar msgIdStr[37];

	uuid_generate_random(msgId);
	uuid_unparse_lower(msgId, msgIdStr);

	objectObj = generate_object_object(privData, meta);

	// root object
	rootObj = json_object_new();
	json_object_set_object_member(rootObj, "object", objectObj);
	//
	rootNode = json_node_new(JSON_NODE_OBJECT);
	json_node_set_object(rootNode, rootObj);

	message = json_to_string(rootNode, TRUE);
	json_node_free(rootNode);
	json_object_unref(rootObj);

	return message;
}

gchar *
generate_mot_event_message(void *privData, NvDsEventMsgMeta *meta)
{
	JsonNode *rootNode;
	JsonObject *rootObj;
	JsonObject *embeddingObj;
	gchar *message;

	uuid_t msgId;
	gchar msgIdStr[37];

	uuid_generate_random(msgId);
	uuid_unparse_lower(msgId, msgIdStr);

	// create root obj
	rootObj = json_object_new();

	// add frame info
	EventMsgSubMeta *msg_meta_content = (EventMsgSubMeta *)meta->extMsg;
	json_object_set_int_member(rootObj, "frame_number", msg_meta_content->frameId);
	json_object_set_int_member(rootObj, "stream_id", msg_meta_content->sensorId);

	// add objects
	JsonArray *jObjectArray = json_array_sized_new(msg_meta_content->num_msg_sub_meta);
	for (size_t i = 0; i < msg_meta_content->num_msg_sub_meta; i++)
	{
		NvDsEventMsgMeta *msg_sub_meta = msg_meta_content->msg_sub_meta_list[i];

		JsonObject *jObject = json_object_new();

		JsonObject *jBoxObject = json_object_new();

		json_object_set_double_member(jBoxObject, "x", msg_sub_meta->bbox.left);
		json_object_set_double_member(jBoxObject, "y", msg_sub_meta->bbox.top);
		json_object_set_double_member(jBoxObject, "w", msg_sub_meta->bbox.width);
		json_object_set_double_member(jBoxObject, "h", msg_sub_meta->bbox.height);
		json_object_set_object_member(jObject, "box", jBoxObject);
		json_object_set_int_member(jObject, "object_id", msg_sub_meta->trackingId);

		JsonArray *jObjectEmbedding = json_array_sized_new(msg_sub_meta->objSignature.size);
		for (size_t j = 0; j < msg_sub_meta->objSignature.size; j++)
			json_array_add_double_element(
				jObjectEmbedding, msg_sub_meta->objSignature.signature[j]);
		json_object_set_array_member(jObject, "embedding", jObjectEmbedding);

		json_array_add_object_element(jObjectArray, jObject);
	}
	json_object_set_array_member(rootObj, "MOT", jObjectArray);

	// create root node
	rootNode = json_node_new(JSON_NODE_OBJECT);
	json_node_set_object(rootNode, rootObj);

	// create message
	message = json_to_string(rootNode, TRUE);
	json_node_free(rootNode);
	json_object_unref(rootObj);

	return message;
}

gchar *generate_XFace_metadata_message(NvDsEventMsgMeta *meta)
{
	JsonNode *rootNode;
	JsonObject *rootObj;
	gchar *message;
	rootObj = json_object_new();

	// add frame info
	XFaceMetaMsg *msg_meta_content = (XFaceMetaMsg *)meta->extMsg;
	// json_object_set_string_member(rootObj, "timestamp", msg_meta_content->timestamp);
	json_object_set_double_member(rootObj, "timestamp", msg_meta_content->timestamp);

	json_object_set_int_member(rootObj, "frame_number", msg_meta_content->frameId);
	json_object_set_string_member(rootObj, "camera_id", g_strdup(msg_meta_content->cameraId));
	json_object_set_string_member(rootObj, "session_id", g_strdup(msg_meta_content->sessionId));

	// add MOT objects
	JsonArray *jMotObjectArray = json_array_sized_new(msg_meta_content->num_mot_obj);
	for (size_t i = 0; i < msg_meta_content->num_mot_obj; i++)
	{
		NvDsMOTMsgData *msg_sub_meta = msg_meta_content->mot_meta_list[i];

		JsonObject *jObject = json_object_new();

		JsonObject *jBoxObject = json_object_new();
		json_object_set_double_member(jBoxObject, "x", msg_sub_meta->bbox.left);
		json_object_set_double_member(jBoxObject, "y", msg_sub_meta->bbox.top);
		json_object_set_double_member(jBoxObject, "w", msg_sub_meta->bbox.width);
		json_object_set_double_member(jBoxObject, "h", msg_sub_meta->bbox.height);
		json_object_set_object_member(jObject, "box", jBoxObject);
		json_object_set_int_member(jObject, "object_id", msg_sub_meta->track_id);

		json_object_set_string_member(jObject, "embedding", msg_sub_meta->embedding);

		json_array_add_object_element(jMotObjectArray, jObject);
	}
	json_object_set_array_member(rootObj, "MOT", jMotObjectArray);

	// add FACE objects
	JsonArray *jFaceObjectArray = json_array_sized_new(msg_meta_content->num_face_obj);
	for (size_t i = 0; i < msg_meta_content->num_face_obj; i++)
	{
		NvDsFaceMsgData *msg_sub_meta = msg_meta_content->face_meta_list[i];

		JsonObject *jObject = json_object_new();

		json_object_set_double_member(jObject, "confidence", msg_sub_meta->confidence_score);
		JsonObject *jBoxObject = json_object_new();
		json_object_set_double_member(jBoxObject, "x", msg_sub_meta->bbox.left);
		json_object_set_double_member(jBoxObject, "y", msg_sub_meta->bbox.top);
		json_object_set_double_member(jBoxObject, "w", msg_sub_meta->bbox.width);
		json_object_set_double_member(jBoxObject, "h", msg_sub_meta->bbox.height);
		json_object_set_object_member(jObject, "box", jBoxObject);

		json_object_set_string_member(jObject, "feature", msg_sub_meta->feature);
		json_object_set_string_member(jObject, "encoded_img", msg_sub_meta->encoded_img);
		json_object_set_string_member(jObject, "staff_id", msg_sub_meta->staff_id);
		json_object_set_string_member(jObject, "name", msg_sub_meta->name);

		json_array_add_object_element(jFaceObjectArray, jObject);
	}
	json_object_set_array_member(rootObj, "FACE", jFaceObjectArray);

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
	gchar *message;
	rootObj = json_object_new();

	// add frame info
	XFaceVisualMsg *msg_meta_content = (XFaceVisualMsg *)meta->extMsg;
	// json_object_set_string_member(rootObj, "timestamp", msg_meta_content->timestamp);
	json_object_set_double_member(rootObj, "timestamp", msg_meta_content->timestamp);

	json_object_set_int_member(rootObj, "frame_number", msg_meta_content->frameId);
	json_object_set_int_member(rootObj, "camera_id", msg_meta_content->cameraId);

	// add MOT objects
	JsonArray *jVisualObjectArray = json_array_sized_new(msg_meta_content->num_cropped_face);
	for (size_t i = 0; i < msg_meta_content->num_cropped_face; i++)
	{
		NvDsVisualMsgData *msg_sub_meta = msg_meta_content->visual_meta_list[i];

		JsonObject *jObject = json_object_new();

		JsonObject *jBoxObject = json_object_new();
		json_object_set_string_member(jObject, "encoded_face_img", msg_sub_meta->cropped_face);
		json_array_add_object_element(jVisualObjectArray, jObject);
	}
	json_object_set_array_member(rootObj, "cropped_face", jVisualObjectArray);

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
		message = generate_XFace_metadata_message(meta);
		break;
	case 2:
		message = generate_XFace_visual_message(meta);
		break;
	default:
		break;
	}
	return message;
}
