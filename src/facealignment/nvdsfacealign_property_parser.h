#ifndef _NVDS_FACEALIGN_PROPERTY_PARSER_H
#define _NVDS_FACEALIGN_PROPERTY_PARSER_H

#include <gst/gst.h>
#include "gstnvfacealign.h"

#define NVDSFACEALIGN_PROPERTY "property"
#define NVDSFACEALIGN_PROPERTY_TARGET_IDS "target-unique-ids"
#define NVDSFACEALIGN_PROPERTY_GPU_ID "gpu-id"
#define NVDSFACEALIGN_PROPERTY_TENSOR_BUF_POOL_SIZE "tensor-buf-pool-size"

#define NVDSFACEALIGN_PROPERTY_NETWORK_SHAPE "network-input-shape"
#define NVDSFACEALIGN_PROPERTY_TENSOR_DATA_TYPE "tensor-data-type"
#define NVDSFACEALIGN_PROPERTY_TENSOR_NAME "tensor-name"

/** Parameters for filtering objects based min/max size threshold. */
#define NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MIN_WIDTH "input-object-min-width"
#define NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MIN_HEIGHT "input-object-min-height"
#define NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MAX_WIDTH "input-object-max-width"
#define NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MAX_HEIGHT "input-object-max-height"

/**
 * @brief config parser
 * 
 * @param nvdsfacealign pointer to GstNvfacealign structure
 * @param cfg_file_path config file path
 * @return gboolean denoting if successfully parsed config file
 */
gboolean nvdsfacealign_parse_config_file (GstNvfacealign *nvdsfacealign, gchar *cfg_file_path);

#endif