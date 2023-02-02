#ifndef APP_STRUCT_H
#define APP_STRUCT_H
#include "params.h"
#include "kafka_producer.h"
#include "tracker.h"
#include "nvds_obj_encode.h"

struct user_callback_data
{
    gchar *session_id;
    std::vector<std::string> video_name;
    KafkaProducer *meta_producer;
    KafkaProducer *visual_producer;
    std::vector<std::shared_ptr<tracker>> trackers;
    gchar *timestamp;
    float face_feature_confidence_threshold;
    float mot_confidence_threshold;
    bool save_crop_img;

    int muxer_output_width;
    int muxer_output_height;
    int muxer_batch_size;
    int muxer_buffer_pool_size;
    int muxer_nvbuf_memory_type;
    bool muxer_live_source;

    int tiler_rows;
    int tiler_cols;
    int tiler_width;
    int tiler_height;

    std::string mot_rawmeta_topic;
    std::string face_rawmeta_topic;
    std::string visual_topic;
    std::string connection_str;
    SinkPerfStruct *fakesink_perf;

    // full frame encode related settings
    int fullframe_encode_scale_width = 640;
    int fullframe_encode_scale_height = 480;
    NvDsObjEncCtxHandle fullframe_ctx_handle;

    std::vector<float> batch_face_feature;
    std::vector<std::shared_ptr<NvDsFaceMsgData>> face_meta_list;
    int frame_since_last_decode_face_name = 0;
};

#endif