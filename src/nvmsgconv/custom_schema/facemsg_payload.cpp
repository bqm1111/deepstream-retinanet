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
		// case NVDS_OBJECT_TYPE_PERSON:
		// 	// person sub object
		// 	jobject = json_object_new();

		// 	if (meta->extMsgSize)
		// 	{
		// 		NvDsPersonObject *dsObj = (NvDsPersonObject *)meta->extMsg;
		// 		if (dsObj)
		// 		{
		// 			json_object_set_double_member(jobject, "confidence", meta->confidence);
		// 		}
		// 	}

		// 	else
		// 	{
		// 		// No person object in meta data. Attach empty person sub object.
		// 		json_object_set_double_member(jobject, "confidence", 1.0);
		// 	}
		// 	json_object_set_object_member(objectObj, "person", jobject);
		// 	break;
		// case NVDS_OBJECT_TYPE_PERSON_EXT:
		// 	// person sub object
		// 	jobject = json_object_new();

		// 	if (meta->extMsgSize)
		// 	{
		// 		NvDsPersonObjectExt *dsObj = (NvDsPersonObjectExt *)meta->extMsg;
		// 		if (dsObj)
		// 		{
		// 			json_object_set_double_member(jobject, "confidence", meta->confidence);
		// 		}
		// 	}
		// 	else
		// 	{
		// 		// No person object in meta data. Attach empty person sub object.
		// 		json_object_set_double_member(jobject, "confidence", 1.0);
		// 	}
		// 	json_object_set_object_member(objectObj, "person", jobject);
		// 	break;
		// case NVDS_OBJECT_TYPE_UNKNOWN:
		// 	if (!meta->objectId)
		// 	{
		// 		break;
		// 	}
		// 	/** No information to add; object type unknown within NvDsEventMsgMeta */
		// 	jobject = json_object_new();
		// 	json_object_set_object_member(objectObj, meta->objectId, jobject);
		// 	break;
		// default:
		// 	cout << "Object type not implemented" << endl;
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
			json_object_set_string_member(jobject, "message", (gchar *)dsObj->feature);
		}
	}
	json_object_set_object_member(objectObj, "message", jobject);
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

	rootNode = json_node_new(JSON_NODE_OBJECT);
	json_node_set_object(rootNode, rootObj);

	message = json_to_string(rootNode, TRUE);
	json_node_free(rootNode);
	json_object_unref(rootObj);

	return message;
}