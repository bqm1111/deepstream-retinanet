#include "FaceBin.h"
#include <algorithm>
#include <cmath>
#include <nvdsinfer_custom_impl.h>
#include <nvds_obj_encode.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <json-glib/json-glib.h>
#include <librdkafka/rdkafkacpp.h>
static void XFace_msg_visual_release_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

    if (srcMeta->extMsgSize > 0)
    {
        XFaceVisualMsg *srcExtMsg = (XFaceVisualMsg *)srcMeta->extMsg;
        g_free(srcExtMsg->cameraId);
        g_free(srcExtMsg->sessionId);
        g_free(srcExtMsg->full_img);

        srcMeta->extMsgSize = 0;
    }
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}
static gchar *generate_XFace_visual_message(NvDsEventMsgMeta *meta)
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

gpointer user_copy_facemark_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(data);
    NvDsFaceMetaData *facemark_meta_data_ptr = reinterpret_cast<NvDsFaceMetaData *>(
        user_meta->user_meta_data);
    NvDsFaceMetaData *new_facemark_meta_data_ptr = reinterpret_cast<NvDsFaceMetaData *>(
        g_memdup(facemark_meta_data_ptr, sizeof(NvDsFaceMetaData)));
    return reinterpret_cast<gpointer>(new_facemark_meta_data_ptr);
}

/**
 * @brief implement NvDsMetaReleaseFun
 *
 * @param data
 * @param user_data
 */
void user_release_facemark_data(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(data);
    NvDsFaceMetaData *facemark_meta_data_ptr = reinterpret_cast<NvDsFaceMetaData *>(
        user_meta->user_meta_data);
    delete facemark_meta_data_ptr;
}

static inline bool cmp(Detection &a, Detection &b)
{
    return a.class_confidence > b.class_confidence;
}

static inline float iou(float lbox[4], float rbox[4])
{
    float interBox[] = {
        std::max(lbox[0], rbox[0]), // left
        std::min(lbox[2], rbox[2]), // right
        std::max(lbox[1], rbox[1]), // top
        std::min(lbox[3], rbox[3]), // bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS /
           ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) +
            (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS + 0.000001f);
}

static void nms(std::vector<Detection> &res, float *output, float post_cluster_thresh = 0.7, float iou_threshold = 0.4)
{
    std::vector<Detection> dets;
    for (int i = 0; i < output[0]; i++)
    {
        if (output[15 * i + 1 + 4] <= post_cluster_thresh)
            continue;
        Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m)
    {
        auto &item = dets[m];
        res.push_back(item);
        // std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n)
        {
            if (iou(item.bbox, dets[n].bbox) > iou_threshold)
            {
                dets.erase(dets.begin() + n);
                --n;
            }
        }
    }
}

static void generate_ts_rfc3339(char *buf, int buf_size)
{
    time_t tloc;
    struct tm tm_log;
    struct timespec ts;
    char strmsec[6]; //.nnnZ\0

    clock_gettime(CLOCK_REALTIME, &ts);
    memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
    gmtime_r(&tloc, &tm_log);
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    int ms = ts.tv_nsec / 1000000;
    g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
    strncat(buf, strmsec, buf_size);
}

static void
generate_event_msg_meta(gpointer data, NvDsObjectMeta *obj_meta)
{
    NvDsFaceMetaData *faceMeta = NULL;
    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
        {
            faceMeta = reinterpret_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);
        }
    }
    NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *)data;
    meta->bbox.top = obj_meta->detector_bbox_info.org_bbox_coords.top;
    meta->bbox.left = obj_meta->detector_bbox_info.org_bbox_coords.left;
    meta->bbox.width = obj_meta->detector_bbox_info.org_bbox_coords.width;
    meta->bbox.height = obj_meta->detector_bbox_info.org_bbox_coords.height;

    meta->ts = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN + 1);
    meta->objectId = (gchar *)g_malloc0(MAX_LABEL_SIZE);

    strncpy(meta->objectId, obj_meta->obj_label, MAX_LABEL_SIZE);
    generate_ts_rfc3339(meta->ts, MAX_TIME_STAMP_LEN);
    if (obj_meta->class_id == FACE_CLASS_ID)
    {
        FaceEventMsgData *obj = (FaceEventMsgData *)g_malloc0(sizeof(FaceEventMsgData));

        obj->feature = g_strdup((gchar *)b64encode(faceMeta->feature, FEATURE_SIZE));
        meta->objType = NVDS_OBJECT_TYPE_FACE;
        meta->extMsg = obj;
        meta->extMsgSize = sizeof(FaceEventMsgData);
    }
}

static gpointer meta_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;
    dstMeta = (NvDsEventMsgMeta *)g_memdup(srcMeta, sizeof(NvDsEventMsgMeta));
    if (srcMeta->ts)
        dstMeta->ts = g_strdup(srcMeta->ts);

    if (srcMeta->objectId)
    {
        dstMeta->objectId = g_strdup(srcMeta->objectId);
    }
    if (srcMeta->extMsgSize > 0)
    {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_FACE)
        {
            FaceEventMsgData *srcObj = (FaceEventMsgData *)srcMeta->extMsg;
            FaceEventMsgData *obj = (FaceEventMsgData *)g_malloc0(sizeof(FaceEventMsgData));
            if (srcObj->feature)
            {
                obj->feature = g_strdup(srcObj->feature);
            }
            dstMeta->extMsg = obj;
            dstMeta->extMsgSize = sizeof(FaceEventMsgData);
        }
    }
    return dstMeta;
}

static void meta_free_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    g_free(srcMeta->ts);
    if (srcMeta->objectId)
    {
        g_free(srcMeta->objectId);
    }
    if (srcMeta->extMsgSize > 0)
    {
        if (srcMeta->objType == NVDS_OBJECT_TYPE_FACE)
        {
            FaceEventMsgData *obj = (FaceEventMsgData *)srcMeta->extMsg;
            if (obj->feature)
            {
                g_free(obj->feature);
            }
        }
        g_free(srcMeta->extMsg);
        srcMeta->extMsgSize = 0;
    }
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}

GstPadProbeReturn NvInferFaceBin::osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GST_ASSERT(buf);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    GST_ASSERT(batch_meta);
    GstElement *tiler = reinterpret_cast<GstElement *>(_udata);
    GST_ASSERT(tiler);
    gint tiler_rows, tiler_cols, tiler_width, tiler_height;
    g_object_get(tiler, "rows", &tiler_rows, "columns", &tiler_cols, "width", &tiler_width, "height", &tiler_height, NULL);

    guint num_rects = 0;
    // loop through each frame in batch
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsMetaList *l_user = NULL;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        float muxer_output_height = frame_meta->pipeline_height;
        float muxer_output_width = frame_meta->pipeline_width;
        // translate from batch_id to the position of this frame in tiler
        int tiler_col = frame_meta->batch_id / tiler_cols;
        int tiler_row = frame_meta->batch_id % tiler_cols;
        int offset_x = tiler_col * tiler_width / tiler_cols;
        int offset_y = tiler_row * tiler_height / tiler_rows;
        // loop through each object in frame data

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
            if (obj_meta->class_id == FACE_CLASS_ID)
            {
                num_rects++;
                // draw landmark here
                for (l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
                {
                    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
                    if (user_meta->base_meta.meta_type != (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
                    {
                        continue;
                    }
                    NvDsFaceMetaData *faceMeta = static_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);
                    NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                    display_meta->num_circles = NUM_FACEMARK;
                    for (int j = 0; j < NUM_FACEMARK; j++)
                    {
                        float x = faceMeta->faceMark[2 * j] * tiler_width / muxer_output_width;
                        float y = faceMeta->faceMark[2 * j + 1] * tiler_height / muxer_output_height;
                        display_meta->circle_params[j].xc = static_cast<unsigned int>(x);
                        display_meta->circle_params[j].yc = static_cast<unsigned int>(y);
                        display_meta->circle_params[j].radius = 4;
                        display_meta->circle_params[j].circle_color.red = 1.0;
                        display_meta->circle_params[j].circle_color.green = 1.0;
                        display_meta->circle_params[j].circle_color.blue = 0.0;
                        display_meta->circle_params[j].circle_color.alpha = 1.0;
                        display_meta->circle_params[j].has_bg_color = 1;
                        display_meta->circle_params[j].bg_color.red = 1.0;
                        display_meta->circle_params[j].bg_color.green = 0.0;
                        display_meta->circle_params[j].bg_color.blue = 0.0;
                        display_meta->circle_params[j].bg_color.alpha = 1.0;
                    }
                    nvds_add_display_meta_to_frame(frame_meta, display_meta);
                }

                // ================== EVENT MESSAGE DATA ========================
                NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));

                generate_event_msg_meta(msg_meta, obj_meta);

                NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool(batch_meta);
                if (user_event_meta)
                {
                    user_event_meta->user_meta_data = (void *)msg_meta;
                    user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
                    user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc)meta_copy_func;
                    user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc)meta_free_func;
                    nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
                }
                else
                {
                    g_print("Error in attaching event meta to buffer\n");
                }
            }
        }

        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        display_meta->num_labels = 1;
        NvOSD_TextParams *txt_params = &display_meta->text_params[0];
        txt_params->display_text = reinterpret_cast<char *>(g_malloc0(MAX_DISPLAY_LEN));
        int offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number = %d Number of faces = %d", frame_meta->frame_num, num_rects);
        // g_print("Frame Number = %d Number of faces = %d\n", frame_meta->frame_num, num_rects);
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }
    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn NvInferFaceBin::pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GstMapInfo inmap = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buf, &inmap, GST_MAP_READ))
    {
        QDTLog::error("%s: %d input buffer mapinfo failed", __FILE__, __LINE__);
        return GST_PAD_PROBE_DROP;
    }
    NvBufSurface *ip_surf = (NvBufSurface *)inmap.data;
    gst_buffer_unmap(buf, &inmap);

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    // FIXME: should parse from config file
    NvDsInferParseDetectionParams detectionParams;
    detectionParams.numClassesConfigured = 1;
    detectionParams.perClassPreclusterThreshold = {0.1};
    detectionParams.perClassPostclusterThreshold = {0.7};

    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsMetaList *l_raw = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        // raw output when enable output-tensor-meta
        float muxer_output_height = frame_meta->pipeline_height;
        float muxer_output_width = frame_meta->pipeline_width;
        for (l_raw = frame_meta->frame_user_meta_list; l_raw != NULL; l_raw = l_raw->next)
        {
            NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_raw->data);
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
            {
                continue;
            }

            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta = reinterpret_cast<NvDsInferTensorMeta *>(user_meta->user_meta_data);
            float pgie_net_height = meta->network_info.height;
            float pgie_net_width = meta->network_info.width;

            VTX_ASSERT(meta->num_output_layers == 1); // we only have one output layer
            for (unsigned int i = 0; i < meta->num_output_layers; i++)
            {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
                if (meta->out_buf_ptrs_dev[i])
                {
                    cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                               info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                }
            }

            /* Parse output tensor, similar to NvDsInferParseCustomRetinaface  */
            std::vector<NvDsInferLayerInfo> outputLayersInfo(meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);
            NvDsInferLayerInfo outputLayerInfo = outputLayersInfo.at(0);
            // std::vector < NvDsInferObjectDetectionInfo > objectList;
            float *output = (float *)outputLayerInfo.buffer;

            std::vector<Detection> res;
            nms(res, output, detectionParams.perClassPostclusterThreshold[0]);

            /* Iterate final rectangules and attach result into frame's obj_meta_list */
            for (const auto &obj : res)
            {
                NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);

                // FIXME: a `meta` can produce more than once `obj`. Hence cannot set obj_meta->unique_component_id = meta->unique_id;
                obj_meta->unique_component_id = meta->unique_id;
                obj_meta->confidence = obj.class_confidence;

                // untracked object. Set tracking_id to -1
                obj_meta->object_id = UNTRACKED_OBJECT_ID;
                obj_meta->class_id = FACE_CLASS_ID; // only have one class

                /* retrieve bouding box */
                float scale_x = muxer_output_width / pgie_net_width;
                float scale_y = muxer_output_height / pgie_net_height;

                obj_meta->detector_bbox_info.org_bbox_coords.left = obj.bbox[0];
                obj_meta->detector_bbox_info.org_bbox_coords.top = obj.bbox[1];
                obj_meta->detector_bbox_info.org_bbox_coords.width = obj.bbox[2] - obj.bbox[0];
                obj_meta->detector_bbox_info.org_bbox_coords.height = obj.bbox[3] - obj.bbox[1];

                obj_meta->detector_bbox_info.org_bbox_coords.left *= scale_x;
                obj_meta->detector_bbox_info.org_bbox_coords.top *= scale_y;
                obj_meta->detector_bbox_info.org_bbox_coords.width *= scale_x;
                obj_meta->detector_bbox_info.org_bbox_coords.height *= scale_y;

                /* for nvdosd */
                NvOSD_RectParams &rect_params = obj_meta->rect_params;
                NvOSD_TextParams &text_params = obj_meta->text_params;
                rect_params.left = obj_meta->detector_bbox_info.org_bbox_coords.left;
                rect_params.top = obj_meta->detector_bbox_info.org_bbox_coords.top;
                rect_params.width = obj_meta->detector_bbox_info.org_bbox_coords.width;
                rect_params.height = obj_meta->detector_bbox_info.org_bbox_coords.height;
                rect_params.border_width = 3;
                rect_params.has_bg_color = 0;
                rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};

                // store landmark in obj_user_meta_list
                NvDsFaceMetaData *face_meta_ptr = new NvDsFaceMetaData();
                for (int j = 0; j < 2 * NUM_FACEMARK; j += 2)
                {
                    face_meta_ptr->stage = NvDsFaceMetaStage::DETECTED;
                    face_meta_ptr->faceMark[j] = obj.landmark[j];
                    face_meta_ptr->faceMark[j + 1] = obj.landmark[j + 1];
                    face_meta_ptr->faceMark[j] *= scale_x;
                    face_meta_ptr->faceMark[j + 1] *= scale_y;
                }

                NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
                user_meta->user_meta_data = static_cast<void *>(face_meta_ptr);
                NvDsMetaType user_meta_type = (NvDsMetaType)NVDS_OBJ_USER_META_FACE;
                user_meta->base_meta.meta_type = user_meta_type;
                user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)user_copy_facemark_meta;
                user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)user_release_facemark_data;
                nvds_add_user_meta_to_obj(obj_meta, user_meta);
                nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);

                NvDsObjEncUsrArgs userData = {0};
                userData.saveImg = FALSE;
                userData.attachUsrMeta = TRUE;
                /* Set if Image scaling Required */
                userData.scaleImg = FALSE;
                userData.scaledWidth = 0;
                userData.scaledHeight = 0;
                /* Quality */
                userData.quality = 80;
                /*Main Function Call */
                nvds_obj_enc_process((NvDsObjEncCtxHandle)_udata, &userData, ip_surf, obj_meta, frame_meta);
            }
        }
    }
    nvds_obj_enc_finish((NvDsObjEncCtxHandle)_udata);

    return GST_PAD_PROBE_OK;
}

static size_t WriteJsonCallback(char *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
}

static gpointer XFace_msg_meta_copy_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = (NvDsEventMsgMeta *)g_memdup(srcMeta, sizeof(NvDsEventMsgMeta));
    dstMeta->extMsg = g_malloc0(sizeof(XFaceMetaMsg));
    XFaceMetaMsg *srcExtMsg = (XFaceMetaMsg *)srcMeta->extMsg;
    XFaceMetaMsg *dstExtMsg = (XFaceMetaMsg *)dstMeta->extMsg;

    dstMeta->componentId = srcMeta->componentId;
    dstExtMsg->cameraId = g_strdup(srcExtMsg->cameraId);
    dstExtMsg->sessionId = g_strdup(srcExtMsg->sessionId);
    dstExtMsg->frameId = srcExtMsg->frameId;
    dstExtMsg->timestamp = srcExtMsg->timestamp;
    dstExtMsg->num_face_obj = srcExtMsg->num_face_obj;
    dstExtMsg->num_mot_obj = srcExtMsg->num_mot_obj;

    dstExtMsg->mot_meta_list = (NvDsMOTMsgData **)g_malloc0(dstExtMsg->num_mot_obj * sizeof(NvDsMOTMsgData *));
    dstExtMsg->face_meta_list = (NvDsFaceMsgData **)g_malloc0(dstExtMsg->num_face_obj * sizeof(NvDsFaceMsgData *));

    // Copy Face
    for (int i = 0; i < dstExtMsg->num_face_obj; i++)
    {
        NvDsFaceMsgData *msg_sub_meta = (NvDsFaceMsgData *)g_malloc0(sizeof(NvDsFaceMsgData));
        msg_sub_meta->bbox.top = srcExtMsg->face_meta_list[i]->bbox.top;
        msg_sub_meta->bbox.left = srcExtMsg->face_meta_list[i]->bbox.left;
        msg_sub_meta->bbox.width = srcExtMsg->face_meta_list[i]->bbox.width;
        msg_sub_meta->bbox.height = srcExtMsg->face_meta_list[i]->bbox.height;

        msg_sub_meta->confidence_score = srcExtMsg->face_meta_list[i]->confidence_score; // confidence score
        msg_sub_meta->feature = g_strdup(srcExtMsg->face_meta_list[i]->feature);         // face feature
        msg_sub_meta->staff_id = g_strdup(srcExtMsg->face_meta_list[i]->staff_id);       // Staff_code
        msg_sub_meta->name = g_strdup(srcExtMsg->face_meta_list[i]->name);               // Staff_code
        msg_sub_meta->encoded_img = g_strdup(srcExtMsg->face_meta_list[i]->encoded_img);

        dstExtMsg->face_meta_list[i] = msg_sub_meta;
    }

    // Copy MOT
    for (int i = 0; i < dstExtMsg->num_mot_obj; i++)
    {
        NvDsMOTMsgData *msg_sub_meta = (NvDsMOTMsgData *)g_malloc0(sizeof(NvDsMOTMsgData));
        msg_sub_meta->bbox.top = srcExtMsg->mot_meta_list[i]->bbox.top;
        msg_sub_meta->bbox.left = srcExtMsg->mot_meta_list[i]->bbox.left;
        msg_sub_meta->bbox.width = srcExtMsg->mot_meta_list[i]->bbox.width;
        msg_sub_meta->bbox.height = srcExtMsg->mot_meta_list[i]->bbox.height;
        msg_sub_meta->track_id = srcExtMsg->mot_meta_list[i]->track_id;

        msg_sub_meta->embedding = g_strdup(srcExtMsg->mot_meta_list[i]->embedding); // person embedding

        dstExtMsg->mot_meta_list[i] = msg_sub_meta;
    }

    dstMeta->extMsgSize = srcMeta->extMsgSize;
    return dstMeta;
}

static void XFace_msg_meta_release_func(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;

    if (srcMeta->extMsgSize > 0)
    {
        // free extMsg content
        XFaceMetaMsg *srcExtMsg = (XFaceMetaMsg *)srcMeta->extMsg;
        // g_free(srcExtMsg->timestamp);
        g_free(srcExtMsg->cameraId);
        g_free(srcExtMsg->sessionId);
        // Delete face

        for (int i = 0; i < srcExtMsg->num_face_obj; i++)
        {
            NvDsFaceMsgData *msg_sub_meta = srcExtMsg->face_meta_list[i];

            g_free(msg_sub_meta->feature);
            g_free(msg_sub_meta->staff_id);
            g_free(msg_sub_meta->name);
            g_free(msg_sub_meta->encoded_img);

            g_free(msg_sub_meta);
        }
        g_free(srcExtMsg->face_meta_list);
        srcExtMsg->num_face_obj = 0;

        // Delete MOT
        for (int i = 0; i < srcExtMsg->num_mot_obj; i++)
        {
            NvDsMOTMsgData *msg_sub_meta = srcExtMsg->mot_meta_list[i];
            g_free(msg_sub_meta->embedding);
            g_free(msg_sub_meta);
        }
        g_free(srcExtMsg->mot_meta_list);
        srcExtMsg->num_mot_obj = 0;
        // free extMsg
        g_free(srcMeta->extMsg);
        // free extMsgSize
        srcMeta->extMsgSize = 0;
    }
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}
void getFaceMetaData(NvDsFrameMeta *frame_meta, NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta, std::vector<NvDsFaceMsgData *> &face_meta_list,
                     user_callback_data *callback_data, NvDsInferLayerInfo *output_layer_info)
{
    NvDsFaceMsgData *face_msg_sub_meta = (NvDsFaceMsgData *)g_malloc0(sizeof(NvDsFaceMsgData));
    face_msg_sub_meta->bbox.top = obj_meta->rect_params.top;
    face_msg_sub_meta->bbox.left = obj_meta->rect_params.left;
    face_msg_sub_meta->bbox.width = obj_meta->rect_params.width;
    face_msg_sub_meta->bbox.height = obj_meta->rect_params.height;

    std::string fileNameString = "crop_img/" + std::to_string(frame_meta->frame_num) + "_" + std::to_string(frame_meta->source_id) +
                                 "_" + std::to_string((int)obj_meta->rect_params.width) + "x" + std::to_string((int)obj_meta->rect_params.height) + ".jpg";

    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        FILE *file;
        if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
        {
            NvDsFaceMetaData *faceMeta = reinterpret_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);

            const int feature_size = output_layer_info->inferDims.numElements;
            float *cur_feature = reinterpret_cast<float *>(output_layer_info->buffer) +
                                 faceMeta->aligned_index * feature_size;
            memcpy(faceMeta->feature, cur_feature, feature_size * sizeof(float));
            face_msg_sub_meta->feature = g_strdup(b64encode(faceMeta->feature, FEATURE_SIZE));

            // Send HTTP request
            CURL *curl = callback_data->curl;
            std::string response_string;
            const char *data = gen_body(1, b64encode(cur_feature, FEATURE_SIZE));
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteJsonCallback);

            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

            // // request over HTTP/2, using the same connection!
            // CURLcode res = curl_easy_perform(curl);

            // switch (res)
            // {
            // case CURLE_OPERATION_TIMEDOUT:
            //     face_msg_sub_meta->confidence_score = 0;
            //     face_msg_sub_meta->staff_id = g_strdup("236573");
            //     face_msg_sub_meta->name= g_strdup("Unknown");

            //     QDTLog::warn("Curl Timed Out. Vectordb server is broken");
            //     continue;
            //     break;
            // case CURLE_OK:
            //     QDTLog::info("Response string = {}", response_string);
            //     break;
            // default:
            //     QDTLog::error("curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            //     break;
            // }
            // std::string response_json = response_string.substr(1, response_string.size() - 2);
            // Document doc;
            // doc.Parse(response_json.c_str());
            // Value &s = doc["distance"];
            // face_msg_sub_meta->confidence_score = s.GetDouble();
            // s = doc["code"];
            // face_msg_sub_meta->staff_id = g_strdup(s.GetString());
            // s = doc["name"];
            // face_msg_sub_meta->name = g_strdup(s.GetString());
        }
        else if (user_meta->base_meta.meta_type == NVDS_CROP_IMAGE_META)
        {
            NvDsObjEncOutParams *enc_jpeg_image =
                (NvDsObjEncOutParams *)user_meta->user_meta_data;
            face_msg_sub_meta->encoded_img = g_strdup(b64encode(enc_jpeg_image->outBuffer, enc_jpeg_image->outLen));

            /* Write to File */
            // file = fopen(fileNameString.c_str(), "wb");
            // fwrite(enc_jpeg_image->outBuffer, sizeof(uint8_t),
            //        enc_jpeg_image->outLen, file);
            // fclose(file);
        }
    }
    face_meta_list.push_back(face_msg_sub_meta);
}

void getMOTMetaData(NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta, std::vector<NvDsMOTMsgData *> &mot_meta_list)
{
    NvDsMOTMsgData *mot_msg_sub_meta = (NvDsMOTMsgData *)g_malloc0(sizeof(NvDsMOTMsgData));
    mot_msg_sub_meta->bbox.top = obj_meta->rect_params.top;
    mot_msg_sub_meta->bbox.left = obj_meta->rect_params.left;
    mot_msg_sub_meta->bbox.width = obj_meta->rect_params.width;
    mot_msg_sub_meta->bbox.height = obj_meta->rect_params.height;
    mot_msg_sub_meta->track_id = obj_meta->object_id;

    for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        if (user_meta->base_meta.meta_type == (NvDsMetaType)NVDS_OBJ_USER_META_MOT)
        {
            NvDsMOTMetaData *motMeta = reinterpret_cast<NvDsMOTMetaData *>(user_meta->user_meta_data);
            mot_msg_sub_meta->embedding = g_strdup(motMeta->feature);
        }
    }
    mot_meta_list.push_back(mot_msg_sub_meta);
}

void NvInferFaceBin::sgie_output_callback(GstBuffer *buf,
                                          NvDsInferNetworkInfo *network_info,
                                          NvDsInferLayerInfo *layers_info,
                                          guint num_layers,
                                          guint batch_size,
                                          gpointer user_data)
{
    user_callback_data *callback_data = reinterpret_cast<user_callback_data *>(user_data);
    /* Find the only output layer */
    NvDsInferLayerInfo *output_layer_info;
    NvDsInferLayerInfo *input_layer_info;
    for (int i = 0; i < num_layers; i++)
    {
        NvDsInferLayerInfo *info = &layers_info[i];
        if (info->isInput)
        {
            input_layer_info = info;
        }
        else
        {
            output_layer_info = info;
            // TODO: the info also include input tensor, which is the 3x112x112 input. COuld be use for something.
        }
    }
    /* Assign feature to NvDsFaceMetaData */
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);

        std::vector<NvDsFaceMsgData *> face_sub_meta_list;
        std::vector<NvDsMOTMsgData *> mot_sub_meta_list;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
            if (obj_meta->class_id == FACE_CLASS_ID)
            {
                getFaceMetaData(frame_meta, batch_meta, obj_meta, face_sub_meta_list, callback_data, output_layer_info);
            }
            else if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
            {
                getMOTMetaData(batch_meta, obj_meta, mot_sub_meta_list);
            }
        }
        // sendFullFrame(surface, batch_meta, frame_meta, callback_data);

        const auto p1 = std::chrono::system_clock::now();

        // ===================================== XFace MetaData sent to Kafka =====================================
        XFaceMetaMsg *msg_meta_content = (XFaceMetaMsg *)g_malloc0(sizeof(XFaceMetaMsg));
        // Get MOT meta
        msg_meta_content->num_mot_obj = mot_sub_meta_list.size();
        msg_meta_content->mot_meta_list = (NvDsMOTMsgData **)g_malloc0(mot_sub_meta_list.size() * sizeof(NvDsMOTMsgData *));
        memcpy(msg_meta_content->mot_meta_list, mot_sub_meta_list.data(), mot_sub_meta_list.size() * sizeof(NvDsMOTMsgData *));

        // Get Face meta
        msg_meta_content->num_face_obj = face_sub_meta_list.size();
        msg_meta_content->face_meta_list = (NvDsFaceMsgData **)g_malloc0(face_sub_meta_list.size() * sizeof(NvDsFaceMsgData *));
        memcpy(msg_meta_content->face_meta_list, face_sub_meta_list.data(), face_sub_meta_list.size() * sizeof(NvDsFaceMsgData *));

        // Generate timestamp
        // msg_meta_content->timestamp = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN + 1);
        // generate_ts_rfc3339(msg_meta_content->timestamp, MAX_TIME_STAMP_LEN);
        msg_meta_content->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(p1.time_since_epoch()).count();
        msg_meta_content->cameraId = g_strdup(std::string(callback_data->video_name[frame_meta->source_id]).c_str());
        msg_meta_content->frameId = frame_meta->frame_num;
        msg_meta_content->sessionId = g_strdup(callback_data->session_id);

        // This is where to create the final NvDsEventMsgMeta before sending
        NvDsEventMsgMeta *meta_msg = (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));
        meta_msg->extMsg = (void *)msg_meta_content;
        meta_msg->extMsgSize = sizeof(XFaceMetaMsg);
        meta_msg->componentId = 1;

        gchar *message = generate_XFaceRawMeta_message(meta_msg);
        RdKafka::ErrorCode err = callback_data->kafka_producer->producer->produce(std::string("RawMeta"),
                                                                                  RdKafka::Topic::PARTITION_UA,
                                                                                  RdKafka::Producer::RK_MSG_COPY,
                                                                                  (gchar *)message,
                                                                                  std::string(message).length(),
                                                                                  NULL, 0,
                                                                                  0, NULL, NULL);
        if (err != RdKafka::ERR_NO_ERROR)
        {
            std::cerr << "% Failed to produce to topic "
                      << ": "
                      << RdKafka::err2str(err) << std::endl;

            if (err == RdKafka::ERR__QUEUE_FULL)
            {
                /* If the internal queue is full, wait for
                 * messages to be delivered and then retry.
                 * The internal queue represents both
                 * messages to be sent and messages that have
                 * been sent or failed, awaiting their
                 * delivery report callback to be called.
                 *
                 * The internal queue is limited by the
                 * configuration property
                 * queue.buffering.max.messages */
                callback_data->kafka_producer->producer->poll(1000 /*block for max 1000ms*/);
            }
        }

        // // Pack EventMsgMeta into UserMeta
        // NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool(batch_meta);
        // if (user_event_meta)
        // {
        //     user_event_meta->user_meta_data = (void *)meta_msg;
        //     user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
        //     user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc)XFace_msg_meta_copy_func;
        //     user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc)XFace_msg_meta_release_func;
        //     nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
        // }
    }
    callback_data->tensor_count++;
}

GstPadProbeReturn NvInferFaceBin::tiler_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata)
{
    GstBuffer *buf = reinterpret_cast<GstBuffer *>(info->data);
    GST_ASSERT(buf);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    GST_ASSERT(batch_meta);

    GstElement *tiler = reinterpret_cast<GstElement *>(_udata);
    GST_ASSERT(tiler);
    gint tiler_rows, tiler_cols, tiler_width, tiler_height = 0;
    g_object_get(tiler, "rows", &tiler_rows, "columns", &tiler_cols, "width", &tiler_width, "height", &tiler_height, NULL);
    assert(tiler_height != 0);

    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsMetaList *l_user = NULL;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
        float muxer_output_height = frame_meta->pipeline_height;
        float muxer_output_width = frame_meta->pipeline_width;
        // translate from batch_id to the position of this frame in tiler
        int tiler_col = frame_meta->batch_id % tiler_cols;
        int tiler_row = frame_meta->batch_id / tiler_cols;
        int offset_x = tiler_col * tiler_width / tiler_cols;
        int offset_y = tiler_row * tiler_height / tiler_rows;
        // g_print("in tiler_sink_pad_buffer_probe batch_id = %d, the tiler offset = %d, %d\n", frame_meta->batch_id, offset_x, offset_y);
        // loop through each object in frame data
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
            if (obj_meta->class_id == FACE_CLASS_ID)
            {
                for (l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
                {
                    NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
                    if (user_meta->base_meta.meta_type != (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
                    {
                        continue;
                    }

                    NvDsFaceMetaData *faceMeta = static_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);
                    // scale the landmark data base on tiler
                    for (int j = 0; j < NUM_FACEMARK; j++)
                    {
                        faceMeta->faceMark[2 * j] = faceMeta->faceMark[2 * j] / tiler_cols + offset_x;
                        faceMeta->faceMark[2 * j + 1] = faceMeta->faceMark[2 * j + 1] / tiler_rows + offset_y;
                    }
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
}
