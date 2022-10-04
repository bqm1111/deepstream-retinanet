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