#include "utils.h"

gchar *b64encode(float *vec, int size)
{
    return g_base64_encode((const guchar *)vec, size * sizeof(float));
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

bool parseJson(std::string filename, std::map<std::string, std::string> &result)
{
    GError *error = NULL;

    // Parse the JSON from the file
    JsonParser *parser = json_parser_new();
    json_parser_load_from_file(parser, filename.c_str(), &error);
    if (error)
    {
        printf("Unable to parse `%s': %s\n", filename.c_str(), error->message);
        g_error_free(error);
        g_object_unref(parser);
        return EXIT_FAILURE;
    }

    // Get the root
    // JsonNode *root = json_parser_get_root(parser);
    JsonReader *reader = json_reader_new(json_parser_get_root(parser));
    // Turn the root into a JSON object
    char **members = json_reader_list_members(reader);
    int i = 0;
    while (members[i] != 0)
    {
        std::string m = members[i];
        json_reader_read_member(reader, members[i]);
        std::string value = json_reader_get_string_value(reader);
        json_reader_end_member(reader);
        printf("parse member %s\n", members[i]);
        printf("parse value %s\n", value.c_str());

        result.insert(std::make_pair(std::string(members[i]), value));
        i++;
    }

    g_strfreev(members);
    g_object_unref(reader);
    g_object_unref(parser);

    return EXIT_SUCCESS;
}