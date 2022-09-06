/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "cuda_runtime.h"
#include "gstnvdsfacealign_allocator.h"

/* Standard GStreamer boiler plate macros */
#define GST_TYPE_NVDSFACEALIGN_ALLOCATOR \
    (gst_nvdsfacealign_allocator_get_type ())
#define GST_NVDSFACEALIGN_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSFACEALIGN_ALLOCATOR,GstNvDsFaceAlignAllocator))
#define GST_NVDSFACEALIGN_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSFACEALIGN_ALLOCATOR,GstNvDsFaceAlignAllocatorClass))
#define GST_IS_NVDSFACEALIGN_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSFACEALIGN_ALLOCATOR))
#define GST_IS_NVDSFACEALIGN_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSFACEALIGN_ALLOCATOR))

typedef struct _GstNvDsFaceAlignAllocator GstNvDsFaceAlignAllocator;
typedef struct _GstNvDsFaceAlignAllocatorClass GstNvDsFaceAlignAllocatorClass;

G_GNUC_INTERNAL GType gst_nvdsfacealign_allocator_get_type (void);

GST_DEBUG_CATEGORY_STATIC (gst_nvdsfacealign_allocator_debug);
#define GST_CAT_DEFAULT gst_nvdsfacealign_allocator_debug

/**
 * Extends the GstAllocator class. Holds the parameters for allocator.
 */
struct _GstNvDsFaceAlignAllocator
{
  /** standard gst allocator */
  GstAllocator allocator;
  /** ID of GPU on which it is being allocated */
  guint gpu_id;
  /** raw buffer size */
  size_t raw_buf_size;
};

/** */
struct _GstNvDsFaceAlignAllocatorClass
{
  /** gst allocator class */
  GstAllocatorClass parent_class;
};

/* Standard boiler plate to create a debug category and initializing the
 * allocator type.
 */
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvdsfacealign_allocator_debug, "facealignallocator", 0, "facealign allocator");
#define gst_nvdsfacealign_allocator_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvDsFaceAlignAllocator, gst_nvdsfacealign_allocator,
    GST_TYPE_ALLOCATOR, _do_init);

/** Type of the memory allocated by the allocator. This can be used to identify
 * buffers / memories allocated by this allocator. */
#define GST_NVDSFACEALIGN_MEMORY_TYPE "facealign"

/** Structure allocated internally by the allocator. */
typedef struct
{
  /** Should be the first member of a structure extending GstMemory. */
  GstMemory mem;
  /** Custom Gst memory for facealign plugin */
  GstNvDsFaceAlignMemory mem_facealign;
} GstNvDsFaceAlignMem;

/* Function called by GStreamer buffer pool to allocate memory using this
 * allocator. */
static GstMemory *
gst_nvdsfacealign_allocator_alloc (GstAllocator * allocator, gsize size,
    GstAllocationParams * params)
{
  GstNvDsFaceAlignAllocator *facealign_allocator = GST_NVDSFACEALIGN_ALLOCATOR (allocator);
  GST_DEBUG_OBJECT(facealign_allocator, "gst_nvdsfacealign_allocator_alloc");
  g_print("gst_nvdsfacealign_allocator_alloc\n");
  GstNvDsFaceAlignMem *nvmem = new GstNvDsFaceAlignMem;
  GstNvDsFaceAlignMemory *tmem = &nvmem->mem_facealign;
  cudaError_t cudaReturn = cudaSuccess;

  cudaReturn = cudaMalloc(&tmem->dev_memory_ptr, facealign_allocator->raw_buf_size);
  if (cudaReturn != cudaSuccess) {
    GST_ERROR ("failed to allocate cuda malloc for tensor with error %s",
        cudaGetErrorName (cudaReturn));
  return nullptr;
  }

  /* Initialize the GStreamer memory structure. */
  gst_memory_init ((GstMemory *) nvmem, (GstMemoryFlags) 0, allocator, nullptr,
      size, params->align, 0, size);

  return (GstMemory *) nvmem;
}

/* Function called by buffer pool for freeing memory using this allocator. */
static void
gst_nvdsfacealign_allocator_free (GstAllocator * allocator, GstMemory * memory)
{
  GstNvDsFaceAlignAllocator *facealign_allocator = GST_NVDSFACEALIGN_ALLOCATOR (allocator);
  GST_DEBUG_OBJECT(facealign_allocator, "gst_nvdsfacealign_allocator_free");
  g_print("gst_nvdsfacealign_allocator_free\n");
  GstNvDsFaceAlignMem *nvmem = (GstNvDsFaceAlignMem *) memory;
  GstNvDsFaceAlignMemory *tmem = &nvmem->mem_facealign;

  cudaFree(tmem->dev_memory_ptr);
  delete nvmem;
  return;
}

/* Function called when mapping memory allocated by this allocator. Should
 * return pointer to GstNvDsFaceAlignMemory. */
static gpointer
gst_nvdsfacealign_memory_map (GstMemory * mem, gsize maxsize, GstMapFlags flags)
{
  GstNvDsFaceAlignMem *nvmem = (GstNvDsFaceAlignMem *) mem;

  return (gpointer) & nvmem->mem_facealign;
}

static void
gst_nvdsfacealign_memory_unmap (GstMemory * mem)
{
}

/* Standard boiler plate. Assigning implemented function pointers. */
static void
gst_nvdsfacealign_allocator_class_init (GstNvDsFaceAlignAllocatorClass * klass)
{
  GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS (klass);

  allocator_class->alloc = GST_DEBUG_FUNCPTR (gst_nvdsfacealign_allocator_alloc);
  allocator_class->free = GST_DEBUG_FUNCPTR (gst_nvdsfacealign_allocator_free);
}

/* Standard boiler plate. Assigning implemented function pointers and setting
 * the memory type. */
static void
gst_nvdsfacealign_allocator_init (GstNvDsFaceAlignAllocator * allocator)
{
  GstAllocator *parent = GST_ALLOCATOR_CAST (allocator);

  GST_DEBUG_OBJECT(allocator, "init");

  parent->mem_type = GST_NVDSFACEALIGN_MEMORY_TYPE;
  parent->mem_map = gst_nvdsfacealign_memory_map;
  parent->mem_unmap = gst_nvdsfacealign_memory_unmap;
}

/* Create a new allocator of type GST_TYPE_NVDSFACEALIGN_ALLOCATOR and initialize
 * members. */
GstAllocator *
gst_nvdsfacealign_allocator_new (size_t raw_buf_size, guint gpu_id)
{
  // g_print("\n%s:%d im in gst_nvdsfacealign_allocator_new\n\n", __FILE__, __LINE__);
  GstNvDsFaceAlignAllocator *allocator = (GstNvDsFaceAlignAllocator *)
      g_object_new (GST_TYPE_NVDSFACEALIGN_ALLOCATOR,
      nullptr);

  GST_DEBUG_OBJECT(allocator, "new");

  allocator->gpu_id = gpu_id;
  allocator->raw_buf_size = raw_buf_size;

  return (GstAllocator *) allocator;
}

GstNvDsFaceAlignMemory *
gst_nvdsfacealign_buffer_get_memory (GstBuffer * buffer)
{
  // g_print("\n%s:%d im in gst_nvdsfacealign_buffer_get_memory\n\n", __FILE__, __LINE__);
  GstMemory *mem;

  mem = gst_buffer_peek_memory (buffer, 0);

  if (!mem || !gst_memory_is_type (mem, GST_NVDSFACEALIGN_MEMORY_TYPE))
  {
    g_print("\n%s:%d gst_memory_is_type memory type is not %s \n\n", __FILE__, __LINE__, GST_NVDSFACEALIGN_MEMORY_TYPE);
    exit(1);
    return nullptr;
  }

  return &(((GstNvDsFaceAlignMem *) mem)->mem_facealign);
}
