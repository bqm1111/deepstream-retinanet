#include "nvdsfacealign_property_parser.h"

GST_DEBUG_CATEGORY (NVDSFACEALIGN_CFG_PARSER_CAT);

#define PARSE_ERROR(details_fmt,...) \
  G_STMT_START { \
    GST_CAT_ERROR (NVDSFACEALIGN_CFG_PARSER_CAT, \
        "Failed to parse config file %s: " details_fmt, \
        cfg_file_path, ##__VA_ARGS__); \
    GST_ELEMENT_ERROR (nvdsfacealign, LIBRARY, SETTINGS, \
        ("Failed to parse config file:%s", cfg_file_path), \
        (details_fmt, ##__VA_ARGS__)); \
    goto done; \
  } G_STMT_END

#define CHECK_ERROR(error, custom_err) \
  G_STMT_START { \
    if (error) { \
      std::string errvalue = "Error while setting property, in group ";  \
      errvalue.append(custom_err); \
      PARSE_ERROR ("%s %s", errvalue.c_str(), error->message); \
    } \
  } G_STMT_END

#define GET_STRING_PROPERTY(group, property, field) {\
  field = g_key_file_get_string(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
}

static gboolean
nvdsfacealign_parse_property_group (GstNvfacealign *nvdsfacealign,
    gchar *cfg_file_path, GKeyFile *key_file, gchar *group)
{
  g_autoptr(GError)error = nullptr;
  gboolean ret = FALSE;
  g_auto(GStrv)keys=nullptr;
  GStrv key=nullptr;
  gint *network_shape_list = nullptr;
  gsize network_shape_list_len = 0;
  gint *target_unique_ids_list = nullptr;
  gsize target_unique_ids_list_len = 0;

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR(error, group);

  for (key = keys; *key; key++){
    if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_GPU_ID)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      nvdsfacealign->gpu_id = val;
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_TENSOR_BUF_POOL_SIZE)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      nvdsfacealign->tensor_buf_pool_size = val;
      
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_NETWORK_SHAPE)) {
      network_shape_list = g_key_file_get_integer_list (key_file, group,*key, &network_shape_list_len, &error);
      if (network_shape_list == nullptr) {
        CHECK_ERROR(error, group);
      }
      nvdsfacealign->tensor_params.network_input_shape.clear();
      for (gsize icnt = 0; icnt < network_shape_list_len; icnt++){
        nvdsfacealign->tensor_params.network_input_shape.push_back(network_shape_list[icnt]);
        GST_CAT_INFO (NVDSFACEALIGN_CFG_PARSER_CAT, "Parsed '%s=%d' in group '%s'",
          *key, network_shape_list[icnt], group);
      }
      g_free(network_shape_list);
      network_shape_list = nullptr;
      nvdsfacealign->property_set.network_input_shape = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_TENSOR_DATA_TYPE)) {
      int val =  g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);

      switch (val) {
        case NvDsDataType_FP32:
          break;
        case NvDsDataType_UINT8:
        case NvDsDataType_INT8:
        case NvDsDataType_UINT32:
        case NvDsDataType_INT32:
        case NvDsDataType_FP16:
          g_printerr ("Error. Not supported value for '%s':'%d'\n",
              NVDSFACEALIGN_PROPERTY_TENSOR_DATA_TYPE, val);
          goto done;
        default:
          g_printerr ("Error. Invalid value for '%s':'%d'\n",
              NVDSFACEALIGN_PROPERTY_TENSOR_DATA_TYPE, val);
          goto done;
      }
      nvdsfacealign->tensor_params.data_type = (NvDsDataType) val;
      nvdsfacealign->property_set.tensor_data_type = TRUE;
    }
    else if (!g_strcmp0(*key, NVDSFACEALIGN_PROPERTY_TENSOR_NAME)) {
      GET_STRING_PROPERTY(group, *key, nvdsfacealign->tensor_params.tensor_name);
      GST_CAT_INFO (NVDSFACEALIGN_CFG_PARSER_CAT, "Parsed %s=%s in group '%s'",
          *key, nvdsfacealign->tensor_params.tensor_name.c_str(), group);
      nvdsfacealign->property_set.tensor_name = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_TENSOR_BUF_POOL_SIZE)) {
      guint val = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR(error, group);
      nvdsfacealign->tensor_buf_pool_size = val;
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_TARGET_IDS)) {
      target_unique_ids_list = g_key_file_get_integer_list (key_file, group,*key, &target_unique_ids_list_len, &error);
      if (target_unique_ids_list == nullptr) {
        CHECK_ERROR(error, group);
      }
      nvdsfacealign->target_unique_ids.clear();
      for (gsize icnt = 0; icnt < target_unique_ids_list_len; icnt++){
        nvdsfacealign->target_unique_ids.push_back(target_unique_ids_list[icnt]);
        GST_CAT_INFO (NVDSFACEALIGN_CFG_PARSER_CAT, "Parsed '%s=%d' in group '%s'",
          *key, target_unique_ids_list[icnt], group);
      }
      g_free(target_unique_ids_list);
      target_unique_ids_list = nullptr;
      nvdsfacealign->property_set.target_unique_ids = TRUE;
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MIN_WIDTH)) {
      nvdsfacealign->min_input_object_width = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      if ((gint) nvdsfacealign->min_input_object_width < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",
            NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MIN_WIDTH,
            nvdsfacealign->min_input_object_width);
        goto done;
      }
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MIN_HEIGHT)) {
      nvdsfacealign->min_input_object_height = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      if ((gint) nvdsfacealign->min_input_object_height < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",
            NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MIN_HEIGHT,
            nvdsfacealign->min_input_object_height);
        goto done;
      }
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MAX_WIDTH)) {
      nvdsfacealign->max_input_object_width = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      if ((gint) nvdsfacealign->max_input_object_width < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",
            NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MAX_WIDTH,
            nvdsfacealign->max_input_object_width);
        goto done;
      }
    }
    else if (!g_strcmp0 (*key, NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MAX_HEIGHT)) {
      nvdsfacealign->max_input_object_height = g_key_file_get_integer (key_file, group, *key, &error);
      CHECK_ERROR (error, group);
      if ((gint) nvdsfacealign->max_input_object_height < 0) {
        g_printerr ("Error: Negative value specified for %s(%d)\n",
            NVDSFACEALIGN_PROPERTY_INPUT_OBJECT_MAX_HEIGHT,
            nvdsfacealign->max_input_object_height);
        goto done;
      }
    }
  }

  if(!(nvdsfacealign->property_set.network_input_shape &&
    nvdsfacealign->property_set.tensor_data_type && 
    nvdsfacealign->property_set.tensor_name &&
    nvdsfacealign->property_set.target_unique_ids)) {
    printf("ERROR: Some preprocess config properties not set\n");
    return FALSE;
  }
  ret = TRUE;

done:
  return ret;
}

gboolean nvdsfacealign_parse_config_file(GstNvfacealign *nvdsfacealign, gchar *cfg_file_path)
{
  g_autoptr(GError) error = nullptr;
  gboolean ret = FALSE;
  g_auto(GStrv)groups=nullptr;
  GStrv group;
  g_autoptr(GKeyFile) cfg_file = g_key_file_new ();

  if (!NVDSFACEALIGN_CFG_PARSER_CAT) {
    GstDebugLevel  level;
    GST_DEBUG_CATEGORY_INIT (NVDSFACEALIGN_CFG_PARSER_CAT, "nvfacealign", 0, NULL);
    level = gst_debug_category_get_threshold (NVDSFACEALIGN_CFG_PARSER_CAT);
    if (level < GST_LEVEL_ERROR )
      gst_debug_category_set_threshold (NVDSFACEALIGN_CFG_PARSER_CAT, GST_LEVEL_ERROR);
  }

  if (!g_key_file_load_from_file (cfg_file, cfg_file_path, G_KEY_FILE_NONE,
          &error)) {
    PARSE_ERROR ("%s", error->message);
  }

  // Check if 'property' group present
  if (!g_key_file_has_group (cfg_file, NVDSFACEALIGN_PROPERTY)) {
    PARSE_ERROR ("Group 'property' not specified");
  }

  groups = g_key_file_get_groups (cfg_file, nullptr);

  for (group = groups; *group; group++) {
    GST_CAT_INFO (NVDSFACEALIGN_CFG_PARSER_CAT, "Group found %s", *group);
    if (!strcmp(*group, NVDSFACEALIGN_PROPERTY)){
      ret = nvdsfacealign_parse_property_group(nvdsfacealign,
          cfg_file_path, cfg_file, *group);
      if (!ret){
        g_print("NVDSPREPROCESS_CFG_PARSER: Group '%s' parse failed\n", *group);
        goto done;
      }
    }
    else {
      g_print("NVDSPREPROCESS_CFG_PARSER: Group '%s' ignored\n", *group);
    }
  }

  g_key_file_set_list_separator (cfg_file,';');


done:
  return ret;
}