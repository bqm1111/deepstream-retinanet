#ifndef COMMON_H_bbbd8c21af5d0e37138de6c4
#define COMMON_H_bbbd8c21af5d0e37138de6c4

#include <gst/gst.h>
#include <cassert>
#include <stdio.h>

#ifndef GST_ASSERT
#define GST_ASSERT(ans) assert_98dae521c1e67e8b70f66d14866fe14e((ans), __FILE__, __LINE__);
inline void assert_98dae521c1e67e8b70f66d14866fe14e(void* element, const char *file, int line)
{
    if (!element) {
        gst_printerr ("could not create element %s:%d\n", file, line);
        gst_object_unref(element);
        exit(-3);
    }
}
#endif // GST_ASSERT

#ifndef VTX_ASSERT
#define VTX_ASSERT assert
#endif

inline gboolean bus_watch_callback(GstBus *_bus, GstMessage *_msg, gpointer _uData)
{
    switch (GST_MESSAGE_TYPE(_msg))
    {
    case GST_MESSAGE_EOS:
        printf("GST_MESSAGE_EOS\r\n");
        break;
    case GST_MESSAGE_WARNING:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_warning(_msg, &error, &debug);
        g_print("Warning: %s: %s\n", error->message, debug);
        g_free(debug);
        g_error_free(error);
        break;
    }
    case GST_MESSAGE_ERROR:
    {
        gchar *debug;
        GError *error;
        gst_message_parse_error(_msg, &error, &debug);
        g_printerr("Error: %s: %s\n", error->message, debug);
        g_free(debug);
        g_error_free(error);
        break;
    }
    default:
        printf(".");
        fflush(stdout);
        break;
    }
    return TRUE;
}

static void newPadCB(GstElement *element, GstPad *pad, gpointer data)
{
    gchar *name;
    name = gst_pad_get_name(pad);

    GstCaps *p_caps = gst_pad_get_pad_template_caps(pad);
    GstElement *sink = GST_ELEMENT(data);
    if (gst_element_link_pads(element, name, sink, "sink") == false)
    {
        gst_print("newPadCB : failed to link elements\n");
        // throw std::runtime_error("");
    }
    g_free(name);
}

struct GstAppParam
{
    GstAppParam()
    {
        muxer_output_height = 1920;
        muxer_output_width = 1080;
        tiler_rows = 1;
        tiler_cols = 2;
        tiler_width = 1280;
        tiler_height = 480;
    }
    int muxer_output_width;
    int muxer_output_height;
    int tiler_rows;
    int tiler_cols;
    int tiler_width;
    int tiler_height;
};
#endif