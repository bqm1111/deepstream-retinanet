#ifndef _GST_NVDS_FACE_ALIGN_ALLOCATOR_H_
#define _GST_NVDS_FACE_ALIGN_ALLOCATOR_H_

#include <cuda_runtime_api.h>
#include <gst/gst.h>
#include <vector>
#include "cudaEGL.h"
#include "nvbufsurface.h"

/**
 * This file describes the custom memory allocator for the Gstreamer TensorRT
 * plugin. The allocator allocates memory for a specified batch_size of frames
 * of resolution equal to the network input resolution and RGBA color format.
 * The frames are allocated on device memory.
 */

/**
 * Holds the pointer for the allocated memory.
 */
typedef struct
{
  /** Pointer to the memory allocated for the batch of frames (DGPU). */
  void *dev_memory_ptr;
} GstNvDsFaceAlignMemory;

/**
 * Get GstNvDsFaceAlignMemory structure associated with buffer allocated using
 * GstNvDsFaceAlignAllocator.
 *
 * @param buffer GstBuffer allocated by this allocator.
 *
 * @return Pointer to the associated GstNvDsFaceAlignMemory structure
 */
GstNvDsFaceAlignMemory *gst_nvdsfacealign_buffer_get_memory (GstBuffer * buffer);

/**
 * Create a new GstNvDsFaceAlignAllocator with the given parameters.
 *
 * @param info video buffer allocator info.
 * @param raw_buf_size size of raw buffer to allocate.
 * @param gpu_id ID of the gpu where the batch memory will be allocated.
 * @param debug_tensor boolean to denote if DEBUG_TENSOR flag is enabled.
 *
 * @return Pointer to the GstNvDsFaceAlignAllocator structure cast as GstAllocator
 */
GstAllocator *gst_nvdsfacealign_allocator_new (size_t raw_buf_size, guint gpu_id);

#endif // _GST_NVDS_FACE_ALIGN_ALLOCATOR_H_