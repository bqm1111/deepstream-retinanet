#include <json-glib/json-glib.h>
#include <uuid.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include "custom_schema.h"
#include "params.h"

gchar *generate_face_event_message(NvDsEventMsgMeta *meta)
{
	JsonNode *rootNode;
	JsonObject *rootObj;
	JsonObject *propObj;
	JsonObject *tsObj;
	JsonObject *cameraIdObj;
	JsonObject *frameIdObj;
	JsonObject *croppedFaceObj;

	gchar *message;
	rootObj = json_object_new();
	propObj = json_object_new();
	tsObj = json_object_new();
	cameraIdObj = json_object_new();
	frameIdObj = json_object_new();
	croppedFaceObj = json_object_new();

	// add frame info
	NvDsFaceMsgData *msg_meta_content = (NvDsFaceMsgData *)meta->extMsg;
	// json_object_set_string_member(rootObj, "timestamp", msg_meta_content->timestamp);
	json_object_set_string_member(rootObj, "title", g_strdup("FaceMeta"));
	json_object_set_string_member(rootObj, "description", g_strdup("Metadata of face detected from video sources"));
	json_object_set_string_member(rootObj, "type", g_strdup("object"));

	// Required
	JsonArray *jFacePropRequired = json_array_sized_new(4);
	json_array_add_string_element(jFacePropRequired, g_strdup("timestamp"));
	json_array_add_string_element(jFacePropRequired, g_strdup("camera_id"));
	json_array_add_string_element(jFacePropRequired, g_strdup("frame_id"));
	json_array_add_string_element(jFacePropRequired, g_strdup("cropped_image"));

	json_object_set_array_member(propObj, "required", jFacePropRequired);

	// timestamp
	json_object_set_string_member(tsObj, "description", g_strdup("Time stamp of the image that this event blong to"));
	json_object_set_string_member(tsObj, "type", g_strdup("double"));
	json_object_set_double_member(tsObj, "value", msg_meta_content->timestamp);

	json_object_set_object_member(propObj, "timestamp", tsObj);

	// Camera_id
	json_object_set_string_member(cameraIdObj, "description", g_strdup("Camera_id of the image that this event blong to"));
	json_object_set_string_member(cameraIdObj, "type", g_strdup("string"));
	json_object_set_string_member(cameraIdObj, "value", g_strdup(msg_meta_content->cameraId));
	json_object_set_object_member(propObj, "camera_id", cameraIdObj);

	// Frame_id
	json_object_set_string_member(frameIdObj, "description", g_strdup("Frame_id of the image that this event blong to"));
	json_object_set_string_member(frameIdObj, "type", g_strdup("integer"));
	json_object_set_int_member(frameIdObj, "value", msg_meta_content->frameId);
	json_object_set_object_member(propObj, "frame_id", frameIdObj);

	// Cropped_image
	json_object_set_string_member(croppedFaceObj, "type", g_strdup("object"));
	JsonObject *facePropObj;
	JsonObject *jObj;
	facePropObj = json_object_new();
	// Required
	jFacePropRequired = json_array_sized_new(9);
	json_array_add_string_element(jFacePropRequired, g_strdup("x"));
	json_array_add_string_element(jFacePropRequired, g_strdup("y"));
	json_array_add_string_element(jFacePropRequired, g_strdup("w"));
	json_array_add_string_element(jFacePropRequired, g_strdup("h"));
	json_array_add_string_element(jFacePropRequired, g_strdup("confidence_score"));
	json_array_add_string_element(jFacePropRequired, g_strdup("name"));
	json_array_add_string_element(jFacePropRequired, g_strdup("staff_id"));
	json_array_add_string_element(jFacePropRequired, g_strdup("feature"));
	json_array_add_string_element(jFacePropRequired, g_strdup("encoded_img"));
	json_object_set_array_member(facePropObj, "required", jFacePropRequired);

	// x
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "top left x coordinate of face image");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->bbox.top);
	json_object_set_object_member(facePropObj, "x", jObj);
	// y
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "top left y coordinate of face image");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->bbox.left);
	json_object_set_object_member(facePropObj, "y", jObj);
	// w
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "width of face image");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->bbox.width);
	json_object_set_object_member(facePropObj, "w", jObj);
	// h
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "height of face image");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->bbox.height);
	json_object_set_object_member(facePropObj, "h", jObj);
	// confidence_score
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "confidence score of name of the person appeared on the face image");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->confidence_score);
	json_object_set_object_member(facePropObj, "confidence_score", jObj);

	// name
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "name of the person appeared on the face image");
	json_object_set_string_member(jObj, "type", "string");
	json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->name));
	json_object_set_object_member(facePropObj, "name", jObj);
	// staff_id
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "staff_id of the person appeared on the face image");
	json_object_set_string_member(jObj, "type", "string");
	json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->staff_id));
	json_object_set_object_member(facePropObj, "staff_id", jObj);

	// feature
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "vector feature of face image");
	json_object_set_string_member(jObj, "type", "bytes");
	json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->feature));
	json_object_set_object_member(facePropObj, "feature", jObj);

	// encoded_img
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "jpeg encoded image of face");
	json_object_set_string_member(jObj, "type", "bytes");
	json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->encoded_img));
	json_object_set_object_member(facePropObj, "encoded_img", jObj);

	json_object_set_object_member(croppedFaceObj, "properties", facePropObj);
	json_object_set_object_member(propObj, "cropped_image", croppedFaceObj);
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
generate_mot_event_message(NvDsEventMsgMeta *meta)
{
	JsonNode *rootNode;
	JsonObject *rootObj;
	JsonObject *propObj;
	JsonObject *tsObj;
	JsonObject *cameraIdObj;
	JsonObject *frameIdObj;
	JsonObject *personBoxObj;

	gchar *message;
	rootObj = json_object_new();
	propObj = json_object_new();
	tsObj = json_object_new();
	cameraIdObj = json_object_new();
	frameIdObj = json_object_new();
	personBoxObj = json_object_new();

	// add frame info
	NvDsMOTMsgData *msg_meta_content = (NvDsMOTMsgData *)meta->extMsg;
	// json_object_set_string_member(rootObj, "timestamp", msg_meta_content->timestamp);
	json_object_set_string_member(rootObj, "title", g_strdup("MOTMeta"));
	json_object_set_string_member(rootObj, "description", g_strdup("Metadata of MOT module"));
	json_object_set_string_member(rootObj, "type", g_strdup("object"));

	// Required
	JsonArray *jFacePropRequired = json_array_sized_new(4);
	json_array_add_string_element(jFacePropRequired, g_strdup("timestamp"));
	json_array_add_string_element(jFacePropRequired, g_strdup("camera_id"));
	json_array_add_string_element(jFacePropRequired, g_strdup("frame_id"));
	json_array_add_string_element(jFacePropRequired, g_strdup("personBox"));

	json_object_set_array_member(propObj, "required", jFacePropRequired);

	// timestamp
	json_object_set_string_member(tsObj, "description", g_strdup("Time stamp of the image that this event blong to"));
	json_object_set_string_member(tsObj, "type", g_strdup("double"));
	json_object_set_double_member(tsObj, "value", msg_meta_content->timestamp);

	json_object_set_object_member(propObj, "timestamp", tsObj);

	// Camera_id
	json_object_set_string_member(cameraIdObj, "description", g_strdup("Camera_id of the image that this event blong to"));
	json_object_set_string_member(cameraIdObj, "type", g_strdup("string"));
	json_object_set_string_member(cameraIdObj, "value", g_strdup(msg_meta_content->cameraId));
	json_object_set_object_member(propObj, "camera_id", cameraIdObj);

	// Frame_id
	json_object_set_string_member(frameIdObj, "description", g_strdup("Frame_id of the image that this event blong to"));
	json_object_set_string_member(frameIdObj, "type", g_strdup("integer"));
	json_object_set_int_member(frameIdObj, "value", msg_meta_content->frameId);
	json_object_set_object_member(propObj, "frame_id", frameIdObj);

	// Cropped_image
	json_object_set_string_member(personBoxObj, "type", g_strdup("object"));
	JsonObject *mainPersonBoxPropObj;
	JsonObject *jObj;
	mainPersonBoxPropObj = json_object_new();
	// Required
	jFacePropRequired = json_array_sized_new(6);
	json_array_add_string_element(jFacePropRequired, g_strdup("x"));
	json_array_add_string_element(jFacePropRequired, g_strdup("y"));
	json_array_add_string_element(jFacePropRequired, g_strdup("w"));
	json_array_add_string_element(jFacePropRequired, g_strdup("h"));
	json_array_add_string_element(jFacePropRequired, g_strdup("track_id"));
	json_array_add_string_element(jFacePropRequired, g_strdup("embedding"));
	json_object_set_array_member(mainPersonBoxPropObj, "required", jFacePropRequired);

	// x
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "top left x coordinate of personBox");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->bbox.top);
	json_object_set_object_member(mainPersonBoxPropObj, "x", jObj);
	// y
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "top left y coordinate of personBox");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->bbox.left);
	json_object_set_object_member(mainPersonBoxPropObj, "y", jObj);
	// w
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "width of personBox");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->bbox.width);
	json_object_set_object_member(mainPersonBoxPropObj, "w", jObj);
	// h
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "height of personBox");
	json_object_set_string_member(jObj, "type", "float");
	json_object_set_double_member(jObj, "value", msg_meta_content->bbox.height);
	json_object_set_object_member(mainPersonBoxPropObj, "h", jObj);
	// track_id
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "track_id of personBox");
	json_object_set_string_member(jObj, "type", "integer");
	json_object_set_int_member(jObj, "value", msg_meta_content->track_id);
	json_object_set_object_member(mainPersonBoxPropObj, "track_id", jObj);

	// embedding
	jObj = json_object_new();
	json_object_set_string_member(jObj, "description", "vector embedding of personBox");
	json_object_set_string_member(jObj, "type", "bytes");
	json_object_set_string_member(jObj, "value", g_strdup(msg_meta_content->embedding));
	json_object_set_object_member(mainPersonBoxPropObj, "embedding", jObj);

	json_object_set_object_member(personBoxObj, "properties", mainPersonBoxPropObj);
	json_object_set_object_member(propObj, "personBox", personBoxObj);
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
gchar *generate_visual_event_message(NvDsEventMsgMeta *meta)
{
	// JsonNode *rootNode;
	// JsonObject *rootObj;
	// gchar *message;
	// rootObj = json_object_new();

	// // add frame info
	// json_object_set_string_member(rootObj, "timestamp", g_strdup("Something here"));


	// // create root node
	// rootNode = json_node_new(JSON_NODE_OBJECT);
	// json_node_set_object(rootNode, rootObj);

	// // create message
	// message = json_to_string(rootNode, TRUE);
	// json_node_free(rootNode);
	// json_object_unref(rootObj);
	return g_strdup("message");
}

gchar *
generate_XFace_event_message(void *privData, NvDsEventMsgMeta *meta)
{
	gchar *message;

	switch (meta->componentId)
	{
	case 1:
	{
		switch (meta->objClassId)
		{
		case FACE_CLASS_ID:
			message = generate_face_event_message(meta);
			break;
		case PGIE_CLASS_ID_PERSON:
			message = generate_mot_event_message(meta);
			break;
		default:
			break;
		}
		break;
	}
	case 2:
	{
		message = generate_visual_event_message(meta);
		break;
	}

	default:
		break;
	}

	return message;
}
