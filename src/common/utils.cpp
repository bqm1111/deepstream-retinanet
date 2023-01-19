#include "utils.h"
#include "QDTLog.h"

gchar *b64encode(float *vec, int size)
{
    return g_base64_encode((const guchar *)vec, size * sizeof(float));
}
gchar *b64encode(uint8_t *vec, int size)
{
    return g_base64_encode((const guchar *)vec, size * sizeof(uint8_t));
}

void floatArr2Str(std::string &str, float *arr, int length)
{
    str = "";
    for (int i = 0; i < length - 1; i++)
    {
        str += std::to_string(arr[i]) + " ";
    }
    str += std::to_string(arr[length]);
}

gchar *gen_body(int num_vec, gchar *vec)
{
    JsonBuilder *builder = json_builder_new();
    builder = json_builder_begin_object(builder);
    builder = json_builder_set_member_name(builder, "num");
    builder = json_builder_add_int_value(builder, num_vec);
    builder = json_builder_set_member_name(builder, "raw");
    builder = json_builder_add_string_value(builder, vec);
    builder = json_builder_end_object(builder);

    JsonGenerator *generator = json_generator_new();
    json_generator_set_root(generator, json_builder_get_root(builder));
    gchar *json = json_generator_to_data(generator, NULL);
    return json;
}

bool parseJson(std::string filename, std::vector<std::string> &name, std::vector<std::vector<std::string>> &info)
{
    std::ifstream ifs{filename};
    if (!ifs.is_open())
    {
        std::cerr << "Could not open file for reading!\n";
        return EXIT_FAILURE;
    }
    IStreamWrapper isw{ifs};

    Document doc{};
    doc.ParseStream(isw);

    for (auto &m : doc.GetObject())
    {
        name.push_back(m.name.GetString());
        auto a = m.value.GetArray();
        std::vector<std::string> value;
        for (Value::ConstValueIterator itr = a.Begin(); itr != a.End(); ++itr)
        {
            value.push_back(itr->GetString());
        }
        info.push_back(value);
    }
    return EXIT_SUCCESS;
}

bool parse_rtsp_src_info(std::string filename, std::vector<std::string> &name, std::vector<std::vector<std::string>> &info)
{
	std::ifstream ifs{filename};
	if (!ifs.is_open())
	{
		std::cerr << "Could not open file for reading!\n";
		return EXIT_FAILURE;
	}
	IStreamWrapper isw{ifs};

	Document doc{};
	doc.ParseStream(isw);

	const Value &content = doc["sources"];

	for (int i = 0; i < content.Size(); i++)
	{
		name.push_back(content[i]["camera_id"].GetString());
		std::vector<std::string> value;
		value.push_back(content[i]["address"].GetString());
		value.push_back(content[i]["encode_type"].GetString());
		value.push_back(content[i]["type"].GetString());
		info.push_back(value);
	}
	return EXIT_SUCCESS;
}

void generate_ts_rfc3339(char *buf, int buf_size)
{
    time_t tloc;
    struct tm tm_log;
    struct timespec ts;
    char strmsec[6];

    clock_gettime(CLOCK_REALTIME, &ts);
    memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
    gmtime_r(&tloc, &tm_log);
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
    int ms = ts.tv_nsec / 1000000;
    g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
    strncat(buf, strmsec, buf_size);
}

std::vector<std::string> parseListJson(std::string response_json)
{
    std::string str = response_json.substr(1, response_json.length() - 2);
    std::vector<std::string> res;
    int first = 0;
    int last = 0;
    for (int i = 0; i < str.size() - 2; i++)
    {
        if (str[i] == '}' && str[i + 1] == ',' && str[i + 2] == '{')
        {
            last = i;
            res.push_back(str.substr(first, last - first + 1));
            first = i + 2;
        }
    }
    res.push_back(str.substr(first, str.length() - first));
    return res;
}

float clip(float x)
{
    x = x > 1 ? 1 : x;
    x = x < 0 ? 0 : x;
    return x;
}

void freeXFaceMOTMsgMeta(XFaceMOTMsgMeta *msg_meta_content)
{
    if (msg_meta_content->timestamp)
        g_free(msg_meta_content->timestamp);
    if (msg_meta_content->sessionId)
        g_free(msg_meta_content->sessionId);
    if (msg_meta_content->cameraId)
        g_free(msg_meta_content->cameraId);

    for (int i = 0; i < msg_meta_content->num_mot_obj; i++)
    {
        if (msg_meta_content->mot_meta_list[i]->embedding)
            g_free(msg_meta_content->mot_meta_list[i]->embedding);
        g_free(msg_meta_content->mot_meta_list[i]);
    }
    g_free(msg_meta_content->mot_meta_list);

    g_free(msg_meta_content);
}

void freeNvDsFaceMsgData(NvDsFaceMsgData *msg_meta_content)
{
    if (msg_meta_content->timestamp)
        g_free(msg_meta_content->timestamp);
    if (msg_meta_content->sessionId)
        g_free(msg_meta_content->sessionId);
    if (msg_meta_content->cameraId)
        g_free(msg_meta_content->cameraId);

    if (msg_meta_content->name)
        g_free(msg_meta_content->name);
    if (msg_meta_content->staff_id)
        g_free(msg_meta_content->staff_id);
    if (msg_meta_content->feature)
        g_free(msg_meta_content->feature);
    if (msg_meta_content->encoded_img)
        g_free(msg_meta_content->encoded_img);

    g_free(msg_meta_content);
}

void freeXFaceVisualMsg(XFaceVisualMsg *msg_meta_content)
{
    if (msg_meta_content->timestamp)
        g_free(msg_meta_content->timestamp);
    if (msg_meta_content->sessionId)
        g_free(msg_meta_content->sessionId);
    if (msg_meta_content->cameraId)
        g_free(msg_meta_content->cameraId);
    if (msg_meta_content->full_img)
        g_free(msg_meta_content->full_img);

    g_free(msg_meta_content);
}
