#ifndef PARAMS_H_8971a2c3da276ee5d7f01820
#define PARAMS_H_8971a2c3da276ee5d7f01820

#include <gst/gst.h>
#include <cassert>
#include <stdio.h>
#include <iostream>
#include <nvdsmeta_schema.h>
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
struct GstAppParam
{
    GstAppParam()
    {
        muxer_output_height = 1080;
        muxer_output_width = 1920;
        tiler_rows = 1;
        tiler_cols = 1;
        tiler_width = 640;
        tiler_height = 480;
    }

    int muxer_output_width;
    int muxer_output_height;
    int tiler_rows;
    int tiler_cols;
    int tiler_width;
    int tiler_height;
};
//
struct CloudParam
{
    CloudParam()
    {
        topic = "face";
        connection_str = "localhost;9092";
        msg_config_path = "../configs/faceid/msgconv_config.txt";
        proto_lib = "src/kafka_protocol_adaptor/libnvds_kafka_proto.so";
        msg2p_lib = "/home/martin/minhbq6/NVIDIA/experiment/build/src/nvmsgconv/libnvmsgconv.so";
        schema_type = NVDS_PAYLOAD_CUSTOM;
        msg2p_meta = 0;
        frame_interval = 30;
    }
    std::string topic;
    std::string connection_str;
    std::string msg_config_path;
    std::string proto_lib;
    std::string msg2p_lib;
    gint schema_type;
    gint msg2p_meta;
    gint frame_interval;
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
    // GstNvDsPreProcessBatchMeta *aligned_tensor = nullptr;
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

#endif