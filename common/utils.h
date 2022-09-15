#ifndef UTILS_H_d6e876b1bc073cc2f2597e6b
#define UTILS_H_d6e876b1bc073cc2f2597e6b

#include <stdio.h>
#include <curl/curl.h>
#include <assert.h>
#include <glib.h>
#include <json-glib/json-glib.h>
#include <string>
gchar* b64encode(float* vec, int size);
void floatArr2Str(std::string &str, float *arr, int length);
gchar* gen_body(int num_vec, gchar* vec);

#endif