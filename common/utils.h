#ifndef UTILS_H_d6e876b1bc073cc2f2597e6b
#define UTILS_H_d6e876b1bc073cc2f2597e6b

#include <stdio.h>
#include <curl/curl.h>
#include <assert.h>
#include <glib.h>
#include <json-glib/json-glib.h>

gchar* b64encode(float* vec, int size);

#endif