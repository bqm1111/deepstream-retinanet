#include "message.h"
gchar *generate_XFace_visual_message(XFaceVisualMsg *msg_meta_content)
{
    JsonNode *rootNode;
    JsonObject *rootObj;
    JsonObject *propObj;

    gchar *message;
    rootObj = json_object_new();
    propObj = json_object_new();

    // add frame info
    json_object_set_string_member(rootObj, "srctime", msg_meta_content->timestamp);
    json_object_set_string_member(rootObj, "camera_id", g_strdup(msg_meta_content->cameraId));
    json_object_set_int_member(rootObj, "frame_id", msg_meta_content->frameId);
    json_object_set_string_member(rootObj, "session_id", msg_meta_content->sessionId);
    json_object_set_int_member(rootObj, "frame_w", msg_meta_content->width);
    json_object_set_int_member(rootObj, "frame_h", msg_meta_content->height);
    json_object_set_string_member(rootObj, "frame", msg_meta_content->full_img);

    // create root node
    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, rootObj);

    // create message
    message = json_to_string(rootNode, TRUE);

    json_node_free(rootNode);
    json_object_unref(rootObj);
    //
    return message;
}

gchar *generate_FaceRawMeta_message(std::shared_ptr<NvDsFaceMsgData> msg_meta_content)
{
    JsonNode *rootNode;
    JsonObject *rootObj;
    JsonObject *propObj;

    gchar *message;
    rootObj = json_object_new();
    propObj = json_object_new();

    // add frame info
    json_object_set_string_member(rootObj, "srctime", msg_meta_content->timestamp);
    json_object_set_string_member(rootObj, "camera_id", g_strdup(msg_meta_content->cameraId));
    json_object_set_int_member(rootObj, "frame_id", msg_meta_content->frameId);
    json_object_set_string_member(rootObj, "session_id", msg_meta_content->sessionId);

    // FACE
    JsonObject *faceObj = json_object_new();
    JsonObject *jbboxObj = json_object_new();
    // y
    json_object_set_double_member(jbboxObj, "y", msg_meta_content->bbox.top);
    // x
    json_object_set_double_member(jbboxObj, "x", msg_meta_content->bbox.left);
    // w
    json_object_set_double_member(jbboxObj, "w", msg_meta_content->bbox.width);
    // h
    json_object_set_double_member(jbboxObj, "h", msg_meta_content->bbox.height);

    json_object_set_object_member(faceObj, "bbox", jbboxObj);

    // confidence_score
    json_object_set_null_member(faceObj, "score");

    // name
    json_object_set_null_member(faceObj, "name");
    // staff_id
    json_object_set_null_member(faceObj, "staff_id");

    // feature
    json_object_set_string_member(faceObj, "feature", g_strdup(msg_meta_content->feature));

    // encoded_img
    json_object_set_string_member(faceObj, "image", g_strdup(msg_meta_content->encoded_img));

    json_object_set_object_member(rootObj, "face", faceObj);

    // create root node
    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, rootObj);

    // create message
    message = json_to_string(rootNode, TRUE);

    json_node_free(rootNode);
    json_object_unref(rootObj);

    return message;
}
gchar *generate_MOTRawMeta_message(XFaceMOTMsgMeta *msg_meta_content)
{
    JsonNode *rootNode;
    JsonObject *rootObj;
    JsonObject *propObj;

    gchar *message;
    rootObj = json_object_new();
    propObj = json_object_new();

    // add frame info
    json_object_set_string_member(rootObj, "srctime", msg_meta_content->timestamp);
    json_object_set_string_member(rootObj, "camera_id", g_strdup(msg_meta_content->cameraId));
    json_object_set_int_member(rootObj, "frame_id", msg_meta_content->frameId);
    json_object_set_string_member(rootObj, "session_id", msg_meta_content->sessionId);

    // MOT
    JsonObject *motArrObj = json_object_new();
    JsonArray *jMOTMetaArray = json_array_sized_new(msg_meta_content->num_mot_obj);
    for (int i = 0; i < msg_meta_content->num_mot_obj; i++)
    {
        JsonObject *motObj = json_object_new();
        JsonObject *jbboxObj = json_object_new();
        // y
        json_object_set_double_member(jbboxObj, "y", msg_meta_content->mot_meta_list[i]->bbox.top);
        // x
        json_object_set_double_member(jbboxObj, "x", msg_meta_content->mot_meta_list[i]->bbox.left);
        // w
        json_object_set_double_member(jbboxObj, "w", msg_meta_content->mot_meta_list[i]->bbox.width);
        // h
        json_object_set_double_member(jbboxObj, "h", msg_meta_content->mot_meta_list[i]->bbox.height);

        json_object_set_object_member(motObj, "bbox", jbboxObj);

        // track_id
        json_object_set_int_member(motObj, "object_id", msg_meta_content->mot_meta_list[i]->track_id);

        // embedding
        json_object_set_string_member(motObj, "embedding", g_strdup(msg_meta_content->mot_meta_list[i]->embedding));

        json_array_add_object_element(jMOTMetaArray, motObj);
    }

    json_object_set_array_member(rootObj, "MOT", jMOTMetaArray);

    // create root node
    rootNode = json_node_new(JSON_NODE_OBJECT);
    json_node_set_object(rootNode, rootObj);

    // create message
    message = json_to_string(rootNode, TRUE);

    json_node_free(rootNode);
    json_object_unref(rootObj);

    return message;
}
