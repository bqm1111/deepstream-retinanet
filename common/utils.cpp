#include "utils.h"

gchar* b64encode(float* vec, int size)
{
    return g_base64_encode((const guchar*)vec, size * sizeof(float));
}
