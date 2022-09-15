#ifndef _NVDS_FACE_ALIGN_INTERFACE_HPP_
#define _NVDS_FACE_ALIGN_INTERFACE_HPP_

#include <vector>
#include <string>

#include "nvbufsurface.h"
#include "params.h"

typedef enum
{
  /** NvDsFaceAlign operation succeeded. */
  NVDSFACEALIGN_SUCCESS = 0,
  /** Failed to configure the tensor_impl instance possibly due to an
     *  erroneous initialization property. */
  NVDSFACEALIGN_CONFIG_FAILED,
  /** Custom Tensor Preparation failed. */
  NVDSFACEALIGN_CUSTOM_TENSOR_FAILED,
  /** Invalid parameters were supplied. */
  NVDSFACEALIGN_INVALID_PARAMS,
  /** Output parsing failed. */
  NVDSFACEALIGN_OUTPUT_PARSING_FAILED,
  /** CUDA error was encountered. */
  NVDSFACEALIGN_CUDA_ERROR,
  /** TensorRT interface failed. */
  NVDSFACEALIGN_TENSORRT_ERROR,
  /** Resource error was encountered. */
  NVDSFACEALIGN_RESOURCE_ERROR,
  /** Tensor Yet not ready to be attached as meta */
  NVDSFACEALIGN_TENSOR_NOT_READY,
} NvDsFaceAlignStatus;

/**
 * Defines model color formats
 */
typedef enum
{
  /** Specifies 24-bit interleaved R-G-B format. */
  NvDsFaceAlignFormat_RGB,
  /** Specifies 24-bit interleaved B-G-R format. */
  NvDsFaceAlignFormat_BGR,
  /** Specifies 8-bit Luma format. */
  NvDsFaceAlignFormat_GRAY,
  /** Specifies 32-bit interleaved R-G-B-A format. */
  NvDsFaceAlignFormat_RGBA,
  /** Specifies 32-bit interleaved B-G-R-x format. */
  NvDsFaceAlignFormat_BGRx,
  /** NCHW planar */
  NvDsFaceAlignFormat_Tensor,
  NvDsFaceAlignFormat_Unknown = 0xFFFFFFFF,
} NvDsFaceAlignFormat;

/**
 * Holds model parameters for tensor preparation
 */
typedef struct
{
  /** Hold the network shape - interpreted based on network input order
   *  For resnet10 : NCHW = infer-batch-size;height;width;num-channels */
  std::vector<int> network_input_shape;
//   /** Holds the network input format. */
//   NvDsFaceAlignFormat network_color_format;
  /** size of tensor buffer */
  guint64 buffer_size = 1;
  /** DataType for tensor formation */
  NvDsDataType data_type;
  /** Memory Type for tensor formation */
  NvBufSurfaceMemType memory_type;
  /** Name of the tensor same as input layer name of model */
  std::string tensor_name;
} NvDsFaceAlignTensorParams;

/**
 * Custom Buffer passed to the custom lib for preparing tensor.
 */
struct NvDsFaceAlignCustomBuf
{
  /** memory ptr where to store prepared tensor */
  void *memory_ptr;
};

/**
 * class for acquiring and releasing a buffer from tensor pool
 * by custom lib.
 */
class NvDsFaceAlignAcquirer
{
public:
  /** method to acquire a buffer from buffer pool */
  virtual NvDsFaceAlignCustomBuf *acquire() = 0;
  /** method to release buffer from buffer pool */
  virtual gboolean release(NvDsFaceAlignCustomBuf *) = 0;
};

/**
 * A preprocess unit for processing which is the aligned face.
 */
typedef struct
{
  /** nvinfer require all NVDS_PREPROCESS_BATCH_META to have the same shape, hence 
   * some unit is just the padding */
  bool is_pad = false;
  /** NvDsObjectParams belonging to the object to be classified. */
  NvDsObjectMeta *obj_meta = nullptr;
  /** NvDsFrameMeta of the frame being preprocessed */
  NvDsFrameMeta *frame_meta = nullptr;
  /** Index of the frame in the batched input GstBuffer. */
  guint batch_index = 0;
  /** The buffer structure the object / frame was converted from. */
  NvBufSurfaceParams *input_surf_params = nullptr;
  /** Pointer to the converted frame memory. This memory contains the frame
   * converted to RGB/RGBA and scaled to network resolution. This memory is
   * given to Output loop as input for mean subtraction and normalization and
   * Tensor Buffer formation for inferencing. */
  // gpointer converted_frame_ptr = nullptr;
  /** New meta for rois provided */
  NvDsRoiMeta roi_meta;

  /** Pointer to the input GstBuffer. */
  GstBuffer *inbuf = nullptr;

  /** hold the result of gst_buffer_map. Might be for DEBUG only */
  NvBufSurface *in_surf = nullptr;

  NvDsFaceMetaData *face_meta = nullptr;
} NvDsFaceAlignUnit;

#endif // _NVDS_FACE_ALIGN_INTERFACE_HPP_