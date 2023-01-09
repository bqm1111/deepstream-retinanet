/* GStreamer
 * Copyright (C) 2022 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_NVFACEALIGN_H_
#define _GST_NVFACEALIGN_H_
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include "nvdsfacealign_interface.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsfacealign_allocator.h"
#include "alignment_logic.h"
#include "params.h"

#ifndef WANTED_SIZE
#define WANTED_SIZE 112
#endif

G_BEGIN_DECLS

#define GST_TYPE_NVFACEALIGN (gst_nvfacealign_get_type())
#define GST_NVFACEALIGN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVFACEALIGN, GstNvfacealign))
#define GST_NVFACEALIGN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVFACEALIGN, GstNvfacealignClass))
#define GST_IS_NVFACEALIGN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVFACEALIGN))
#define GST_IS_NVFACEALIGN_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVFACEALIGN))

typedef struct _GstNvfacealign GstNvfacealign;
typedef struct _GstNvfacealignClass GstNvfacealignClass;

/** Used by plugin to access GstBuffer and GstNvDsPreProcessMemory
 *  acquired by Custom Library */
struct NvDsFaceAlignCustomBufImpl : public  NvDsFaceAlignCustomBuf
{
  /** Gst Buffer acquired from gst allocator */
  GstBuffer *gstbuf;
  /** Memory corresponding to the gst buffer */
  GstNvDsFaceAlignMemory *memory;
};

/**
 *  For Acquiring/releasing buffer from buffer pool
 */
class NvDsFaceAlignAcquirerImpl : public NvDsFaceAlignAcquirer
{
public:
  /** constructor */
  NvDsFaceAlignAcquirerImpl(GstBufferPool *pool);
  /** override acquire method in plugin */
  NvDsFaceAlignCustomBuf* acquire() override;
  /** override release method in plugin */
  gboolean release(NvDsFaceAlignCustomBuf *) override;

private:
  GstBufferPool *m_gstpool = nullptr;
};

/**
 *  struct denoting properties set by config file
 */
typedef struct {
  /** for config param : target-unique-ids */
  gboolean target_unique_ids;
  /** for config param : network-input-shape */
  gboolean network_input_shape;
  /** for config param : tensor-data-type */
  gboolean tensor_data_type;
  /** for config param : tensor-name */
  gboolean tensor_name;
} NvDsFaceAlignPropertySet;

struct _GstNvfacealign
{
  GstBaseTransform base_nvfacealign;

  /** Target unique ids */
  std::vector <guint64> target_unique_ids;

  /** GPU ID on which we expect to execute the task */
  guint gpu_id;

  /** Boolean to signal output thread to stop. */
  gboolean stop;

  /** struct denoting properties set by config file */
  NvDsFaceAlignPropertySet property_set;

  /** Internal buffer pool for memory required for scaling input frames and
    * cropping object. */
  GstBufferPool *scaling_pool;

  /** Config file path for nvdspreprocess **/
  gchar *config_file_path;

  /** Config file parsing status **/
  gboolean config_file_parse_successful;

  /** where the alignment actually happend.
   * In the preprocess example. this function is implemented in another lib
   */
  std::unique_ptr<AlignmentLogic> logic;

  /** Class for acquiring/releasing buffer from tensor pool */
  std::unique_ptr <NvDsFaceAlignAcquirerImpl> acquire_impl;

  /** pointer to buffer provided to custom library for tensor preparation */
  NvDsFaceAlignCustomBuf *tensor_buf;

  /** tensor buffer pool size */
  guint tensor_buf_pool_size;

  /** Parameters for tensor preparation */
  NvDsFaceAlignTensorParams tensor_params;

  /** Boolean indicating if the bound buffer contents should be written to file. */
  gboolean write_raw_buffers_to_file;
  
  /** Host memory to copy output tensor */
  gpointer h_tensor;

  /** Internal buffer pool for memory required for tensor preparation */
  GstBufferPool *tensor_pool;

  /** GstFlowReturn returned by the latest buffer pad push. */
  GstFlowReturn last_flow_ret;

  /** Input object size-based filtering parameters for object processing mode. */
  guint min_input_object_width;
  guint min_input_object_height;
  guint max_input_object_width;
  guint max_input_object_height;
};

struct _GstNvfacealignClass
{
  GstBaseTransformClass base_nvfacealign_class;
};

GType gst_nvfacealign_get_type(void);

G_END_DECLS

#endif
