#ifndef FACE_DEFINE_H
#define FACE_DEFINE_H

#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h" // must bellow gstnvdsmeta.h
#include "gstnvdsinfer.h"        // must bellow gstnvdsmeta.h

#ifndef FACE_GST_ASSERT
#define FACE_GST_ASSERT(ans) faceid::assert_ddf5d59dd639a351afee3df715b804d2((ans), __FILE__, __LINE__);
#endif

#ifndef VTX_ASSERT
#include <cassert>
#define VTX_ASSERT assert
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

    inline void assert_ddf5d59dd639a351afee3df715b804d2(void *element, const char *file, int line)
    {
        if (!element)
        {
            gst_printerr("could not create element %s:%d\n", file, line);
            gst_object_unref(element);
            exit(-3);
        }
    }

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
    
    /* form in the post process of detection */
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
        const char* name;
        int staff_id;
        float naming_score;
    };

#endif