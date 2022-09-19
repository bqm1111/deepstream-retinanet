#include "faceApp.h"
#include <thread>
#include "ConfigManager.h"
#include "DeepStreamAppConfig.h"
#include "QDTLog.h"
#include "utils.h"
static GMainLoop *loop = NULL;
static gboolean event_thread_func(gpointer arg)
{
	int c = fgetc(stdin);
	switch (c)
	{
	case 'q':
		g_main_loop_quit(loop);
		break;

	default:
		break;
	}
}

int main(int argc, char *argv[])
{
	QDTLog::init();

	GstBus *bus = NULL;
	guint bus_watch_id;

	gst_init(&argc, &argv);
	loop = g_main_loop_new(NULL, FALSE);

	FaceApp app;
	app.loadConfig("../configs/DsApp.conf");

	app.create("face-app");
	app.addVideoSource("../configs/video_list.json");

	// app.showVideo();
	// app.faceDetection();

	// app.detectAndSend();
	app.MOT();
	bus = gst_pipeline_get_bus(GST_PIPELINE(app.getPipeline()));
	GST_ASSERT(bus);
	bus_watch_id = gst_bus_add_watch(bus, bus_watch_callback, nullptr);

	gst_element_set_state(app.getPipeline(), GST_STATE_PLAYING);
	g_timeout_add(40, event_thread_func, NULL);

	g_main_loop_run(loop);

	gst_element_set_state(app.getPipeline(), GST_STATE_NULL);
	g_source_remove(bus_watch_id);
	g_main_loop_unref(loop);
	return 0;

	// std::vector<std::string> name;
	// std::map<std::string, std::string> res;
	// parseJson("../configs/video_list.json", name, res);

	// for(const auto &n:name)
	// {
	// 	std::cout << n << std::endl;
	// }
	// for (const auto &entry : res)
	// {
	// 	std::cout << "{" << entry.first << ", " << entry.second << "}" << std::endl;
	// }
}
