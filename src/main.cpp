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
	QDTLog::init("logs/");

	GstBus *bus = NULL;
	guint bus_watch_id;

	gst_init(&argc, &argv);
	loop = g_main_loop_new(NULL, FALSE);

	FaceApp app;
	app.loadConfig("../configs/DsApp.conf");

	app.create("face-app");
	app.setLive(std::stoi(argv[2]));
	app.addVideoSource(std::string(argv[1]));
	// app.detect();
	// app.MOT();
	// app.detectAndMOT();
	app.sequentialDetectAndMOT();
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
}
// #include "rapidjson/document.h"
// #include "rapidjson/writer.h"
// #include "rapidjson/stringbuffer.h"
// #include <iostream>

// using namespace rapidjson;

// int main()
// {
// 	// 1. Parse a JSON string into DOM.
// 	std::string json = std::string("[{\"distance\":0.5077230334281921,\"code\":\"000000\",\"phone\":\"0123456789\",\"email\":\"000000@example.com\",\"name\":\"Unknown\"}]");
// 	std::string trim = json.substr(1, json.size() - 2);
// 	std::cout << trim <<std::endl;
// 	// const char* json = "{\"project\":\"rapidjson\",\"stars\":\"minh\"}";
// 	Document d;
// 	d.Parse(trim.c_str());

// 	// // 2. Modify it by DOM.
// 	// Value& s = d["stars"];
// 	// s.SetInt(s.GetInt() + 1);

// 	Value&s = d["distance"];
// 	std::cout << s.GetDouble() << std::endl;

// 	return 0;
// }
