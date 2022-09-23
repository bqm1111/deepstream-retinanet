#ifndef PARAMS_H_8971a2c3da276ee5d7f01820
#define PARAMS_H_8971a2c3da276ee5d7f01820

#include <gst/gst.h>
#include <cassert>
#include <stdio.h>
#include <iostream>
#include <nvdsmeta_schema.h>
#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h" // must bellow gstnvdsmeta.h
#include "gstnvdsinfer.h"        // must bellow gstnvdsmeta.h

#ifndef NVDS_OBJ_USER_META_MOT
#define NVDS_OBJ_USER_META_MOT (nvds_get_user_meta_type("NVIDIA.NVINFER.OBJ_USER_META_MOT"))
#endif

#ifndef NVDS_OBJ_USER_META_FACE
#define NVDS_OBJ_USER_META_FACE (nvds_get_user_meta_type("NVIDIA.NVINFER.OBJ_USER_META_FACE"))
#endif

#ifndef NUM_FACEMARK
#define NUM_FACEMARK 5
#endif

#ifndef FACE_CLASS_ID
#define FACE_CLASS_ID 1000
#endif

#ifndef FEATURE_SIZE
#define FEATURE_SIZE 512
#endif

#ifndef MAX_DISPLAY_LEN
#define MAX_DISPLAY_LEN 64
#endif
#define MAX_TIME_STAMP_LEN 32
#define PGIE_CLASS_ID_VEHICLE 2
#define PGIE_CLASS_ID_PERSON 0

#ifndef FACEID_PGIE_CONFIG_PATH
#define FACEID_PGIE_CONFIG_PATH "../configs/faceid/faceid_primary.txt"
#endif
#ifndef FACEID_ALIGN_CONFIG_PATH
#define FACEID_ALIGN_CONFIG_PATH "../configs/faceid/faceid_align_config.txt"
#endif
#ifndef FACEID_SGIE_CONFIG_PATH
#define FACEID_SGIE_CONFIG_PATH "../configs/faceid/faceid_secondary.txt"
#endif

#ifndef MOT_PGIE_CONFIG_PATH
#define MOT_PGIE_CONFIG_PATH "../configs/faceid/mot_primary.txt"
#endif

#ifndef MOT_SGIE_CONFIG_PATH
#define MOT_SGIE_CONFIG_PATH "../configs/faceid/mot_sgie.txt"
#endif

#ifndef MSG_CONFIG_PATH
#define MSG_CONFIG_PATH "../configs/faceid/msgconv_config.txt"
#endif

#ifndef KAFKA_MSG2P_LIB
#define KAFKA_MSG2P_LIB "src/nvmsgconv/libnvmsgconv.so"
#endif

#ifndef KAFKA_PROTO_LIB
#define KAFKA_PROTO_LIB "src/kafka_protocol_adaptor/libnvds_kafka_proto.so"
#endif 

#define POST_TRACK_SCORE 1.0

struct GstAppParam
{
    int muxer_output_width;
    int muxer_output_height;
    int tiler_rows;
    int tiler_cols;
    int tiler_width;
    int tiler_height;

    std::string topic;
    std::string connection_str;
    std::string curl_address;
};

struct alignas(float) Detection
{
    float bbox[4]; // x1 y1 x2 y2
    float class_confidence;
    float landmark[10];
};

// typedef NvDsFaceAlignMeta NvDsFacePointsMetaData;

struct FacePose
{
    float yaw, pitch, roll;
};
enum NvDsFaceMetaStage
{
    EMPTY = -1,
    DETECTED = 0,
    DROPPED,
    ALIGNED,
    MASKPOSED,
    FEATURED,
    NAMED,
};

struct NvDsFaceMetaData
{
    NvDsFaceMetaStage stage;

    /* Assigned in detection */
    float faceMark[2 * NUM_FACEMARK];

    /* Assigned in maskpose */
    bool hasMask;
    FacePose pose;

    /* Assigned in alignment.
     * This face will be formed into the aligned_index in aligned_tensor
     */
    GstNvDsPreProcessBatchMeta *aligned_tensor = nullptr;
    int aligned_index;

    /* Assigned in feature extraction */
    float feature[FEATURE_SIZE];

    /* Assigned in the naming process */
    // std::string name = "";
    // void* customFaceMeta = nullptr;
    const char *name;
    int staff_id;
    float naming_score;
};

typedef struct FaceEventMsgData
{
    gchar *feature;
} FaceEventMsgData;

enum EventMsgSubMetaType
{
    SGIE_EVENT,
    TRACKER_EVENT
};

struct EventMsgSubMeta
{
    EventMsgSubMetaType type;
    gint frameId;
    gint sensorId;
    guint num_msg_sub_meta;
    NvDsEventMsgMeta **msg_sub_meta_list;
};

struct XFaceMsgMeta
{
    double timestamp;
    gint frameId;
    gint cameraId;
    gint num_face_obj;
    gint num_mot_obj;
    NvDsEventMsgMeta **face_meta_list;
    NvDsEventMsgMeta **mot_meta_list;
};

typedef struct NvDsMOTMetaData
{
    gchar *feature;
} NvDsMOTMetaData;


#endif