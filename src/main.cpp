#include <gst/video/video-frame.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <gst/gst.h>
#include <glib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../samples/sample.h"
#include "faceApp.h"
#include <thread>

int main(int argc, char *argv[])
{
	// test_mp4(argc, argv);
	// infer_meta(argc, argv);
	// user_metadata(argc, argv);
	GMainLoop *loop = NULL;
	GstBus *bus = NULL;
	guint bus_watch_id;

	gst_init(&argc, &argv);
	loop = g_main_loop_new(NULL, FALSE);

	FaceApp app("face-app");
	for (int i = 1; i < argc; i++)
	{
		app.add_video(argv[i], "video-" + std::to_string(i));
	}
	// 
	// app.showVideo();
	// app.faceDetection();

	app.detectAndSend();
	bus = gst_pipeline_get_bus(GST_PIPELINE(app.getPipeline()));
	GST_ASSERT(bus);
	bus_watch_id = gst_bus_add_watch(bus, bus_watch_callback, nullptr);

	gst_element_set_state(app.getPipeline(), GST_STATE_PLAYING);
	g_main_loop_run(loop);

	gst_element_set_state(app.getPipeline(), GST_STATE_NULL);
	g_source_remove(bus_watch_id);
	g_main_loop_unref(loop);
	return 0;
}
