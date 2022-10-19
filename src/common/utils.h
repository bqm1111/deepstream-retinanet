#ifndef UTILS_H_d6e876b1bc073cc2f2597e6b
#define UTILS_H_d6e876b1bc073cc2f2597e6b

#include <stdio.h>
#include <curl/curl.h>
#include <assert.h>
#include <glib.h>
#include <json-glib/json-glib.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <iostream>

using namespace rapidjson;
gchar *b64encode(float *vec, int size);
gchar *b64encode(uint8_t *vec, int size);
void floatArr2Str(std::string &str, float *arr, int length);
gchar *gen_body(int num_vec, gchar *vec);
bool parseJson(std::string filename, std::vector<std::string> &name,
               std::vector<std::vector<std::string>> &info);
void generate_ts_rfc3339(char *buf, int buf_size);

#endif
