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
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstnvfacealign
 *
 * The nvfacealign element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v fakesrc ! nvfacealign ! FIXME ! fakesink
 * ]|
 * FIXME Describe what the pipeline does.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <memory>

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include "gstnvfacealign.h"

#include "gstnvdsfacealign_allocator.h"
#include "alignment_logic.h"
#include "nvdsfacealign_property_parser.h"

GST_DEBUG_CATEGORY_STATIC(gst_nvfacealign_debug_category);
#define GST_CAT_DEFAULT gst_nvfacealign_debug_category

/* boiler plate prototypes */

static void gst_nvfacealign_set_property(GObject *object,
                                         guint property_id, const GValue *value, GParamSpec *pspec);
static void gst_nvfacealign_get_property(GObject *object,
                                         guint property_id, GValue *value, GParamSpec *pspec);
static gboolean gst_nvfacealign_set_caps(GstBaseTransform *trans,
                                         GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_nvfacealign_start(GstBaseTransform *trans);
static gboolean gst_nvfacealign_stop(GstBaseTransform *trans);

/* my prototypes */
static GstFlowReturn gst_nvdsfacealign_submit_input_buffer(GstBaseTransform *btrans,
                                                           gboolean discont, GstBuffer *inbuf);
static GstFlowReturn gst_nvdsfacealign_generate_output(GstBaseTransform *btrans, GstBuffer **outbuf);

enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_GPU_DEVICE_ID,
  PROP_CONFIG_FILE,
  PROP_OUTPUT_WRITE_TO_FILE
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_GPU_ID 0
#define DEFAULT_CONFIG_FILE_PATH ""
#define DEFAULT_OUTPUT_WRITE_TO_FILE FALSE
#define DEFAULT_TENSOR_BUF_POOL_SIZE 6 /** Tensor Buffer Pool Size */

/* pad templates */
/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvfacealign_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
                            GST_PAD_SRC,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_nvfacealign_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
                            GST_PAD_SINK,
                            GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

/* class initialization */

G_DEFINE_TYPE_WITH_CODE(GstNvfacealign, gst_nvfacealign, GST_TYPE_BASE_TRANSFORM,
                        GST_DEBUG_CATEGORY_INIT(gst_nvfacealign_debug_category, "nvfacealign", 0,
                                                "debug category for nvfacealign element"));

static void
gst_nvfacealign_class_init(GstNvfacealignClass *klass)
{
  // Indicates we want to use DS buf api
  g_setenv("DS_NEW_BUFAPI", "1", TRUE);

  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                            &gst_nvfacealign_src_template);
  gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass),
                                            &gst_nvfacealign_sink_template);

  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
                                        "FIXME Long name", "Generic", "FIXME Description",
                                        "FIXME <fixme@example.com>");

  gobject_class->set_property = gst_nvfacealign_set_property;
  gobject_class->get_property = gst_nvfacealign_get_property;
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_nvfacealign_set_caps);
  base_transform_class->start = GST_DEBUG_FUNCPTR(gst_nvfacealign_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_nvfacealign_stop);

  /* custom */
  base_transform_class->submit_input_buffer = GST_DEBUG_FUNCPTR(gst_nvdsfacealign_submit_input_buffer);
  base_transform_class->generate_output = GST_DEBUG_FUNCPTR(gst_nvdsfacealign_generate_output);

  /* Install properties */
  g_object_class_install_property(gobject_class, PROP_UNIQUE_ID,
                                  g_param_spec_uint("unique-id",
                                                    "Unique ID",
                                                    "Unique ID for the element. Can be used to identify output of the"
                                                    " element",
                                                    0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(gobject_class, PROP_GPU_DEVICE_ID,
                                  g_param_spec_uint("gpu-id",
                                                    "Set GPU Device ID",
                                                    "Set GPU Device ID", 0,
                                                    G_MAXUINT, DEFAULT_GPU_ID,
                                                    GParamFlags(G_PARAM_READWRITE |
                                                                G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));
  g_object_class_install_property(gobject_class, PROP_CONFIG_FILE,
                                  g_param_spec_string("config-file-path", "Face Alignment Config File",
                                                      "Face Alignment Config File",
                                                      DEFAULT_CONFIG_FILE_PATH,
                                                      (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(gobject_class, PROP_OUTPUT_WRITE_TO_FILE,
                                  g_param_spec_boolean("raw-output-file-write", "Raw Output File Write",
                                                       "Write raw inference output to file",
                                                       DEFAULT_OUTPUT_WRITE_TO_FILE,
                                                       (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                                                                     GST_PARAM_MUTABLE_READY)));
}

static void
gst_nvfacealign_init(GstNvfacealign *nvfacealign)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM(nvfacealign);

  GST_DEBUG_OBJECT(nvfacealign, "init");

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(btrans), TRUE);

  nvfacealign->gpu_id = DEFAULT_GPU_ID;
  nvfacealign->config_file_parse_successful = FALSE;
  nvfacealign->tensor_buf_pool_size = DEFAULT_TENSOR_BUF_POOL_SIZE;

  // int raw_batch = 4;
  // nvfacealign->tensor_params.network_input_shape = std::vector<int>{raw_batch, 3, WANTED_SIZE, WANTED_SIZE};
  // nvfacealign->tensor_params.data_type = NvDsDataType_FP32;
  // FIXME: memory_type for tensor is always NVBUF_MEM_CUDA_DEVICE?
  nvfacealign->tensor_params.memory_type = NVBUF_MEM_CUDA_PINNED;
  // nvfacealign->tensor_params.tensor_name = "data";
}

void gst_nvfacealign_set_property(GObject *object, guint property_id,
                                  const GValue *value, GParamSpec *pspec)
{
  GstNvfacealign *nvfacealign = GST_NVFACEALIGN(object);

  GST_DEBUG_OBJECT(nvfacealign, "set_property");

  switch (property_id)
  {
  case PROP_UNIQUE_ID:
    // nvfacealign->unique_id = g_value_get_uint (value);
    break;
  case PROP_GPU_DEVICE_ID:
    nvfacealign->gpu_id = g_value_get_uint(value);
    break;
  case PROP_OUTPUT_WRITE_TO_FILE:
    nvfacealign->write_raw_buffers_to_file = g_value_get_boolean(value);
    break;
  case PROP_CONFIG_FILE:
  {
    g_free(nvfacealign->config_file_path);
    nvfacealign->config_file_path = g_value_dup_string(value);
    /* Parse the initialization parameters from the config file. This function
     * gives preference to values set through the set_property function over
     * the values set in the config file. */
    nvfacealign->config_file_parse_successful =
        nvdsfacealign_parse_config_file(nvfacealign,
                                        nvfacealign->config_file_path);
    if (nvfacealign->config_file_parse_successful)
    {
      GST_DEBUG_OBJECT(nvfacealign, "Successfully Parsed Config file");
    }
  }
  break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    break;
  }
}

void gst_nvfacealign_get_property(GObject *object, guint property_id,
                                  GValue *value, GParamSpec *pspec)
{
  GstNvfacealign *nvfacealign = GST_NVFACEALIGN(object);

  GST_DEBUG_OBJECT(nvfacealign, "get_property");

  switch (property_id)
  {
  case PROP_UNIQUE_ID:
    // g_value_set_uint (value, nvfacealign->unique_id);
    break;
  case PROP_GPU_DEVICE_ID:
    g_value_set_uint(value, nvfacealign->gpu_id);
    break;
  case PROP_CONFIG_FILE:
    g_value_set_string(value, nvfacealign->config_file_path);
    break;
  case PROP_OUTPUT_WRITE_TO_FILE:
    g_value_set_boolean(value, nvfacealign->write_raw_buffers_to_file);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
    break;
  }
}

static gboolean
gst_nvfacealign_set_caps(GstBaseTransform *trans, GstCaps *incaps,
                         GstCaps *outcaps)
{
  GstNvfacealign *nvfacealign = GST_NVFACEALIGN(trans);

  GST_DEBUG_OBJECT(nvfacealign, "set_caps");

  return TRUE;
}

/* states */
static gboolean
gst_nvfacealign_start(GstBaseTransform *trans)
{
  GstNvfacealign *nvfacealign = GST_NVFACEALIGN(trans);

  GST_DEBUG_OBJECT(nvfacealign, "start");

  if (!nvfacealign->config_file_path || strlen(nvfacealign->config_file_path) == 0)
  {
    GST_ELEMENT_ERROR(nvfacealign, LIBRARY, SETTINGS,
                      ("Configuration file not provided"), (nullptr));
    return FALSE;
  }

  if (nvfacealign->config_file_parse_successful == FALSE)
  {
    GST_ELEMENT_ERROR(nvfacealign, LIBRARY, SETTINGS,
                      ("Configuration file parsing failed"),
                      ("Config file path: %s", nvfacealign->config_file_path));
    return FALSE;
  }

  /**
   * TODO: allocate buffer pool for aligned faces at start and keeps reusing them
   *
   */
  GstStructure *tensor_pool_config;
  GstAllocator *tensor_pool_allocator;
  GstAllocationParams tensor_pool_allocation_params;
  cudaError_t cudaReturn;

  nvfacealign->tensor_pool = gst_buffer_pool_new();
  tensor_pool_config = gst_buffer_pool_get_config(nvfacealign->tensor_pool);
  gst_buffer_pool_config_set_params(tensor_pool_config, nullptr,
                                    sizeof(GstNvDsFaceAlignMemory), nvfacealign->tensor_buf_pool_size,
                                    nvfacealign->tensor_buf_pool_size);

  nvfacealign->tensor_params.buffer_size = 1;
  for (auto &p : nvfacealign->tensor_params.network_input_shape)
  {
    nvfacealign->tensor_params.buffer_size *= p;
  }

  switch (nvfacealign->tensor_params.data_type)
  {
  case NvDsDataType_FP32:
  case NvDsDataType_UINT32:
  case NvDsDataType_INT32:
    nvfacealign->tensor_params.buffer_size *= 4;
    break;
  case NvDsDataType_UINT8:
  case NvDsDataType_INT8:
    nvfacealign->tensor_params.buffer_size *= 1;
    break;
  case NvDsDataType_FP16:
    nvfacealign->tensor_params.buffer_size *= 2;
    break;
  default:
    GST_ELEMENT_ERROR(nvfacealign, LIBRARY, SETTINGS,
                      ("Tensor data type : %d is not Supported\n",
                       (int)nvfacealign->tensor_params.data_type),
                      (nullptr));
    goto error;
  }

  /** NOTE: allocate nvfacealign->tensor_params.buffer_size size. It's good! **/
  tensor_pool_allocator = gst_nvdsfacealign_allocator_new(nvfacealign->tensor_params.buffer_size, nvfacealign->gpu_id);
  GST_DEBUG_OBJECT(nvfacealign, "Allocating Tensor Buffer Pool with size = %d data-type=%d", nvfacealign->tensor_buf_pool_size,
                   nvfacealign->tensor_params.data_type);
  memset(&tensor_pool_allocation_params, 0, sizeof(tensor_pool_allocation_params));
  gst_buffer_pool_config_set_allocator(tensor_pool_config, tensor_pool_allocator,
                                       &tensor_pool_allocation_params);

  if (!gst_buffer_pool_set_config(nvfacealign->tensor_pool, tensor_pool_config))
  {
    GST_ELEMENT_ERROR(nvfacealign, RESOURCE, FAILED,
                      ("Failed to set config on tensor buffer pool"), (nullptr));
    goto error;
  }

  /* Start the buffer pool and allocate all internal buffers. */
  if (!gst_buffer_pool_set_active(nvfacealign->tensor_pool, TRUE))
  {
    GST_ELEMENT_ERROR(nvfacealign, RESOURCE, FAILED,
                      ("Failed to set tensor buffer pool to active"), (nullptr));
    goto error;
  }

  cudaReturn = cudaSetDevice(nvfacealign->gpu_id);
  if (cudaReturn != cudaSuccess)
  {
    GST_ELEMENT_ERROR(nvfacealign, RESOURCE, FAILED,
                      ("Failed to set cuda device %d", nvfacealign->gpu_id),
                      ("cudaSetDevice failed with error %s", cudaGetErrorName(cudaReturn)));
    goto error;
  }

  /** class for acquiring/releasing buffer from tensor pool */
  nvfacealign->acquire_impl = std::make_unique<NvDsFaceAlignAcquirerImpl>(nvfacealign->tensor_pool);
  nvfacealign->tensor_buf = new NvDsFaceAlignCustomBuf;

  /** class for actually do the alignment */
  nvfacealign->logic = std::make_unique<AlignmentLogic>(nvfacealign->tensor_params);

  /** memory to save */
  if (nvfacealign->write_raw_buffers_to_file)
    nvfacealign->h_tensor = g_malloc(nvfacealign->tensor_params.buffer_size);

  return TRUE;

error:
  // delete[] nvdspreprocess->transform_params.src_rect;
  // delete[] nvdspreprocess->transform_params.dst_rect;

  // delete[] nvdspreprocess->batch_insurf.surfaceList;
  // delete[] nvdspreprocess->batch_outsurf.surfaceList;

  // if (nvdspreprocess->convert_stream) {
  //   cudaStreamDestroy (nvdspreprocess->convert_stream);
  //   nvdspreprocess->convert_stream = NULL;
  // }

  return FALSE;
}

static gboolean
gst_nvfacealign_stop(GstBaseTransform *trans)
{
  GstNvfacealign *nvfacealign = GST_NVFACEALIGN(trans);

  GST_DEBUG_OBJECT(nvfacealign, "stop");

  nvfacealign->acquire_impl.reset();

  if (nvfacealign->tensor_buf)
  {
    delete nvfacealign->tensor_buf;
    nvfacealign->tensor_buf = NULL;
  }

  gst_object_unref(nvfacealign->tensor_pool);

  if (nvfacealign->h_tensor)
  {
    g_free(nvfacealign->h_tensor);
  }

  return TRUE;
}

static void
release_user_meta_at_batch_level(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  GstNvDsPreProcessBatchMeta *preprocess_batchmeta = (GstNvDsPreProcessBatchMeta *)user_meta->user_meta_data;
  if (preprocess_batchmeta->tensor_meta != nullptr)
  {
    gst_buffer_unref((GstBuffer *)preprocess_batchmeta->tensor_meta->private_data); // unref tensor pool buffer
    delete preprocess_batchmeta->tensor_meta;
  }
  if (preprocess_batchmeta->private_data != nullptr)
    gst_buffer_unref((GstBuffer *)preprocess_batchmeta->private_data); // unref conversion pool buffer

  for (auto &roi_meta : preprocess_batchmeta->roi_vector)
  {
    g_list_free(roi_meta.classifier_meta_list);
    g_list_free(roi_meta.roi_user_meta_list);
  }

  delete preprocess_batchmeta;
}

/* NOTE: nvinfer require all NVDS_PREPROCESS_BATCH_META to have the same shape
 * if all.size() <= batch_size. Return the single batch
 * if all.size() % batch_size == 0:
 * if all.size() / batch_size != 0: add pad unit
 */

std::vector<std::vector<NvDsFaceAlignUnit>> batch_divide(std::vector<NvDsFaceAlignUnit> all, unsigned int batch_size)
{
  std::vector<std::vector<NvDsFaceAlignUnit>> ret;
  for (unsigned int i = 0; i < all.size(); i += batch_size)
  {
    std::vector<NvDsFaceAlignUnit> aBatch;
    for (int j = i; j < i + batch_size; j++)
    {
      if (j < all.size())
        aBatch.push_back(all.at(j));
      else
      {
        /* create fake unit */
        NvDsFaceAlignUnit unit;
        unit.is_pad = true;
        // unit.roi_meta.frame_meta = frame_meta;
        unit.roi_meta.scale_ratio_x = 1.0;
        unit.roi_meta.scale_ratio_y = 1.0;
        unit.roi_meta.offset_left = 0.0;
        unit.roi_meta.offset_top = 0.0;
        unit.roi_meta.classifier_meta_list = g_list_alloc();
        unit.roi_meta.roi_user_meta_list = g_list_alloc();
        aBatch.push_back(unit);
      }
    }
    ret.push_back(aBatch);
  }
  return ret;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_nvdsfacealign_submit_input_buffer(GstBaseTransform *btrans,
                                      gboolean discont, GstBuffer *inbuf)
{
  GstNvfacealign *nvfacealign = GST_NVFACEALIGN(btrans);
  GstFlowReturn flow_ret = GST_FLOW_ERROR;

  nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(nvfacealign));

  if (FALSE == nvfacealign->config_file_parse_successful)
  {
    GST_ELEMENT_ERROR(nvfacealign, LIBRARY, SETTINGS,
                      ("Configuration file parsing failed\n"),
                      ("Config file path: %s\n", nvfacealign->config_file_path));
    return flow_ret;
  }

  // TODO: do the alignment here, use inbuf as input
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
  if (batch_meta == nullptr)
  {
    GST_ELEMENT_ERROR(nvfacealign, STREAM, FAILED,
                      ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  GstMapInfo in_map_info = GST_MAP_INFO_INIT;
  if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ))
  {
    GST_ELEMENT_ERROR(nvfacealign, STREAM, FAILED,
                      ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  NvBufSurface *in_surf = (NvBufSurface *)in_map_info.data;
  if (!in_surf)
  {
    GST_ELEMENT_ERROR(nvfacealign, STREAM, FAILED,
                      ("%s:in_map_info.data failed", __func__), (NULL));
    return GST_FLOW_ERROR;
  }

  std::vector<NvDsFaceAlignUnit> units;
  /**
   * NOTE: Access the frame buffer, access the landmark and produce the tensor
   */
  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = reinterpret_cast<NvDsFrameMeta *>(l_frame->data);
    if (!frame_meta)
    {
      GST_ERROR_OBJECT(nvfacealign, "not found NvDsFrameMeta in batch_meta");
      flow_ret = GST_FLOW_ERROR;
      gst_buffer_unmap(inbuf, &in_map_info);
      return flow_ret;
    }
    gint source_id = frame_meta->source_id;  /* source id of incoming buffer */
    gint batch_index = frame_meta->batch_id; /* batch id of incoming buffer */

    // NOTE: access width height of frame by
    // in_surf->surfaceList[batch_index].width;
    std::vector<NvDsObjectMeta *> to_remove;
    for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = reinterpret_cast<NvDsObjectMeta *>(l_obj->data);
      if (obj_meta->class_id != FACE_CLASS_ID)
        continue;

      for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next)
      {
        NvDsUserMeta *user_meta = reinterpret_cast<NvDsUserMeta *>(l_user->data);
        if (user_meta->base_meta.meta_type != (NvDsMetaType)NVDS_OBJ_USER_META_FACE)
          continue;

        NvDsFaceMetaData *face_meta = static_cast<NvDsFaceMetaData *>(user_meta->user_meta_data);

        if (face_meta->stage > NvDsFaceMetaStage::ALIGNED)
          continue;

        if (obj_meta->detector_bbox_info.org_bbox_coords.width >= nvfacealign->min_input_object_width &&
            obj_meta->detector_bbox_info.org_bbox_coords.width <= nvfacealign->max_input_object_width &&
            obj_meta->detector_bbox_info.org_bbox_coords.height >= nvfacealign->min_input_object_height &&
            obj_meta->detector_bbox_info.org_bbox_coords.height <= nvfacealign->max_input_object_height)
        {
          /* NOTE: the following attributes of unit are mandatory */
          NvDsFaceAlignUnit unit;
          unit.frame_meta = frame_meta;
          unit.obj_meta = obj_meta;
          unit.face_meta = face_meta;
          unit.batch_index = batch_index;
          unit.input_surf_params = in_surf->surfaceList + batch_index; // input image
          unit.inbuf = inbuf;
          unit.in_surf = in_surf;
          unit.roi_meta.frame_meta = frame_meta;
          unit.roi_meta.scale_ratio_x = 1.0;
          unit.roi_meta.scale_ratio_y = 1.0;
          unit.roi_meta.offset_left = 0.0;
          unit.roi_meta.offset_top = 0.0;
          unit.roi_meta.classifier_meta_list = g_list_alloc();
          unit.roi_meta.roi_user_meta_list = g_list_alloc();
          unit.roi_meta.roi.top = obj_meta->detector_bbox_info.org_bbox_coords.top;
          unit.roi_meta.roi.left = obj_meta->detector_bbox_info.org_bbox_coords.left;
          unit.roi_meta.roi.width = obj_meta->detector_bbox_info.org_bbox_coords.width;
          unit.roi_meta.roi.height = obj_meta->detector_bbox_info.org_bbox_coords.height;
          units.push_back(unit);
        }
        else
        {
          to_remove.push_back(obj_meta);
        }
      }
    }
    for (NvDsObjectMeta *obj_meta : to_remove)
    {
      nvds_remove_obj_meta_from_frame(frame_meta, obj_meta);
    }
  }

  /* devide all the units into batches, each batches has the maxium element of nvfacealign->tensor_params.network_input_shape[0] */
  std::vector<std::vector<NvDsFaceAlignUnit>> batched_unit;
  {
    int max_batch_size = nvfacealign->tensor_params.network_input_shape[0];
    bool require_padding = false;
    if (units.size() > max_batch_size && units.size() % max_batch_size != 0)
      require_padding = true;

    NvDsFrameMeta *fake_frame_meta;
    if (require_padding)
    {
      fake_frame_meta = nvds_acquire_frame_meta_from_pool(batch_meta);
      fake_frame_meta->batch_id = 1000;
      fake_frame_meta->frame_num = 0;
    }

    for (unsigned int i = 0; i < units.size(); i += max_batch_size)
    {
      std::vector<NvDsFaceAlignUnit> aBatch;
      for (int j = i; j < i + max_batch_size; j++)
      {
        if (j < units.size())
          aBatch.push_back(units.at(j));
        else if (require_padding)
        {
          /* create fake unit */
          NvDsFaceAlignUnit unit;
          unit.is_pad = true;
          unit.roi_meta.frame_meta = fake_frame_meta;
          unit.roi_meta.scale_ratio_x = 1.0;
          unit.roi_meta.scale_ratio_y = 1.0;
          unit.roi_meta.offset_left = 0.0;
          unit.roi_meta.offset_top = 0.0;
          unit.roi_meta.classifier_meta_list = g_list_alloc();
          unit.roi_meta.roi_user_meta_list = g_list_alloc();
          aBatch.push_back(unit);
        }
      }
      batched_unit.push_back(aBatch);
    }
  }

  /* NOTE: nvinfer require all NVDS_PREPROCESS_BATCH_META to have the same shape
   * hence if there are more than one NVDS_PREPROCESS_BATCH_META, we should use the maxium input batch
   * if not, we should use the real input batch name to get better efficiency
   * */
  std::vector<int> tensor_shape = nvfacealign->tensor_params.network_input_shape;
  guint64 buffer_size = nvfacealign->tensor_params.buffer_size;
  if (batched_unit.size() == 1 && batched_unit.end()->size() != nvfacealign->tensor_params.network_input_shape[0])
  {
    tensor_shape[0] = batched_unit[0].size();
    buffer_size = 1;
    for (auto &p : tensor_shape)
    {
      buffer_size *= p;
    }
    switch (nvfacealign->tensor_params.data_type)
    {
    case NvDsDataType_FP32:
    case NvDsDataType_UINT32:
    case NvDsDataType_INT32:
      buffer_size *= 4;
      break;
    case NvDsDataType_UINT8:
    case NvDsDataType_INT8:
      buffer_size *= 1;
      break;
    case NvDsDataType_FP16:
      buffer_size *= 2;
      break;
    default:
      GST_ELEMENT_ERROR(nvfacealign, LIBRARY, SETTINGS,
                        ("Tensor data type : %d is not Supported\n",
                         (int)nvfacealign->tensor_params.data_type),
                        (nullptr));
      exit(1);
    }
  }

  if (batched_unit.size() > 1)
  {
    // g_print("\n\n\n\n %s:%d about to attach more than once NVDS_PREPROCESS_BATCH_META to a batch_meta\n\n\n\n", __FILE__, __LINE__);
  }

  for (auto &abatch : batched_unit)
  {
    nvfacealign->tensor_buf = nvfacealign->acquire_impl.get()->acquire();
    GST_DEBUG_OBJECT(nvfacealign, "acquire a tensor memory from pool at addresss %p",
                     ((NvDsFaceAlignCustomBufImpl *)nvfacealign->tensor_buf)->memory->dev_memory_ptr);

    if (NVDSFACEALIGN_SUCCESS != nvfacealign->logic->CustomTensorPreparation(abatch, nvfacealign->tensor_buf))
    {
      GST_ELEMENT_ERROR(nvfacealign, STREAM, FAILED,
                        ("Internal data stream error."),
                        ("CustomTensorPreparation fault"));
      nvfacealign->acquire_impl.get()->release(nvfacealign->tensor_buf);
      goto error;
    };

    GstNvDsPreProcessBatchMeta *preprocess_batchmeta = new GstNvDsPreProcessBatchMeta();
    preprocess_batchmeta->target_unique_ids = nvfacealign->target_unique_ids;
    /* FIXME: each object must have a roi_vector. With `use_raw_input=true`, it seems that
     * the roi_vector do nothing. But the size of roi_vector must match the batch of the raw input tensor
     */
    for (int i = 0; i < abatch.size(); i++)
    {
      preprocess_batchmeta->roi_vector.push_back(abatch[i].roi_meta);
      if (abatch[i].is_pad)
      {
        continue;
      }
      abatch[i].face_meta->stage = NvDsFaceMetaStage::ALIGNED;
      abatch[i].face_meta->aligned_index = i;
      /* NOTE: also refer to this tensor in NVDS_OBJ_USER_META_FACE. Easy to access later. */
      abatch[i].face_meta->aligned_tensor = preprocess_batchmeta;
    }

    GST_DEBUG_OBJECT(nvfacealign, "%s:%d preprocess_batchmeta->roi_vector size %d", __FILE__, __LINE__, preprocess_batchmeta->roi_vector.size());
    preprocess_batchmeta->tensor_meta = new NvDsPreProcessTensorMeta();
    preprocess_batchmeta->tensor_meta->gpu_id = nvfacealign->gpu_id;
    preprocess_batchmeta->private_data = nullptr; // scaling pool
    preprocess_batchmeta->tensor_meta->private_data =
        ((NvDsFaceAlignCustomBufImpl *)nvfacealign->tensor_buf)->gstbuf;
    preprocess_batchmeta->tensor_meta->raw_tensor_buffer =
        ((NvDsFaceAlignCustomBufImpl *)nvfacealign->tensor_buf)->memory->dev_memory_ptr;
    preprocess_batchmeta->tensor_meta->buffer_size = buffer_size; // size of the tensor
    preprocess_batchmeta->tensor_meta->tensor_shape = tensor_shape;
    preprocess_batchmeta->tensor_meta->data_type = nvfacealign->tensor_params.data_type;
    preprocess_batchmeta->tensor_meta->tensor_name = nvfacealign->tensor_params.tensor_name;

    NvDsUserMeta *batch_user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    batch_user_meta->user_meta_data = preprocess_batchmeta;
    batch_user_meta->base_meta.meta_type = NVDS_PREPROCESS_BATCH_META;
    batch_user_meta->base_meta.copy_func = NULL;
    batch_user_meta->base_meta.release_func = release_user_meta_at_batch_level;
    batch_user_meta->base_meta.batch_meta = batch_meta;

    if (nvfacealign->write_raw_buffers_to_file)
    {
      g_assert(cudaSuccess == cudaMemcpy(nvfacealign->h_tensor, preprocess_batchmeta->tensor_meta->raw_tensor_buffer, buffer_size, cudaMemcpyDeviceToHost));

      // frame-number_stream-number_object-number_object-type_widthxheight.jpg
      // gstnvfacealign_batchtensor_object-number%p.bin
      const int MAX_STR_LEN = 1024;
      char *file_name = (char *)g_malloc0(MAX_STR_LEN);
      g_snprintf(file_name, MAX_STR_LEN - 1, "gstnvfacealign_batchtensor_%ld_%p.bin", std::time(0), preprocess_batchmeta->tensor_meta->raw_tensor_buffer);

      FILE *file = fopen(file_name, "w");
      if (!file)
      {
        g_printerr("Could not open file '%s' for writing:%s\n",
                   file_name, strerror(errno));
      }
      else
      {
        fwrite(nvfacealign->h_tensor,
               sizeof(float),
               tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3],
               file);
      }
      g_free(file_name);
      fclose(file);
    }

    nvds_add_user_meta_to_batch(batch_meta, batch_user_meta);
    GST_DEBUG_OBJECT(nvfacealign, "attached preprocessed tensor with shape (%d, %d, %d, %d) at %p to batch_meta=%p",
                     preprocess_batchmeta->tensor_meta->tensor_shape[0], preprocess_batchmeta->tensor_meta->tensor_shape[1],
                     preprocess_batchmeta->tensor_meta->tensor_shape[2], preprocess_batchmeta->tensor_meta->tensor_shape[3],
                     ((NvDsFaceAlignCustomBufImpl *)nvfacealign->tensor_buf)->memory->dev_memory_ptr, batch_meta);
  }
  
  /**
   * NOTE: Push no matter how many tensor attached
   */
  nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(nvfacealign));
  flow_ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(nvfacealign), inbuf);
  if (flow_ret < GST_FLOW_OK)
  {
    // Signal the application for pad push errors by posting a error message on the pipeline bus.
    GST_ELEMENT_ERROR(nvfacealign, STREAM, FAILED,
                      ("Internal data stream error."),
                      ("streaming stopped, reason %s (%d)",
                       gst_flow_get_name(flow_ret), flow_ret));
    goto error;
  }
  else
  {
    GST_DEBUG_OBJECT(nvfacealign, "pushed to downstream");
  }
  if (nvfacealign->last_flow_ret != flow_ret)
    nvfacealign->last_flow_ret = flow_ret;

  return GST_FLOW_OK;

error:
  gst_buffer_unmap(inbuf, &in_map_info);
  return flow_ret;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_nvdsfacealign_generate_output(GstBaseTransform *btrans, GstBuffer **outbuf)
{
  GstNvfacealign *nvfacealign = GST_NVFACEALIGN(btrans);
  return nvfacealign->last_flow_ret;
}

NvDsFaceAlignAcquirerImpl::NvDsFaceAlignAcquirerImpl(GstBufferPool *pool)
{
  m_gstpool = pool;
}

NvDsFaceAlignCustomBuf *NvDsFaceAlignAcquirerImpl::acquire()
{
  GstBuffer *gstbuf;
  GstNvDsFaceAlignMemory *memory;
  GstFlowReturn flow_ret;

  flow_ret = gst_buffer_pool_acquire_buffer(m_gstpool, &gstbuf, nullptr);
  if (flow_ret != GST_FLOW_OK)
  {
    GST_ERROR("error while acquiring buffer from tensor pool\n");
    return nullptr;
  }

  memory = gst_nvdsfacealign_buffer_get_memory(gstbuf);
  if (!memory)
  {
    GST_ERROR("error while getting memory from tensor pool\n");
    return nullptr;
  }

  // NOTE: Ep kieu (NvDsFaceAlignCustomBuf) NvDsFaceAlignCustomBufImpl
  // memory->dev_memory_ptr and memory is actually the same
  // call gstbuf = nullptr after use the memory. the NvDsFaceAlignAcquirerImpl::release function did it correctly
  return new NvDsFaceAlignCustomBufImpl{{memory->dev_memory_ptr}, gstbuf, memory};
}

gboolean NvDsFaceAlignAcquirerImpl::release(NvDsFaceAlignCustomBuf *buf)
{
  NvDsFaceAlignCustomBufImpl *implBuf = (NvDsFaceAlignCustomBufImpl *)(buf);
  gst_buffer_unref(implBuf->gstbuf);
  delete implBuf;
  return TRUE;
}

static gboolean
plugin_init(GstPlugin *plugin)
{

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register(plugin, "nvfacealign", GST_RANK_NONE,
                              GST_TYPE_NVFACEALIGN);
}

/* FIXME: these are normally defined by the GStreamer build system.
   If you are creating an element to be included in gst-plugins-*,
   remove these, as they're always defined.  Otherwise, edit as
   appropriate for your external plugin package. */
#ifndef VERSION
#define VERSION "0.0.FIXME"
#endif
#ifndef PACKAGE
#define PACKAGE "FIXME_package"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "FIXME_package_name"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://FIXME.org/"
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  nvfacealign,
                  "FIXME plugin description",
                  plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)