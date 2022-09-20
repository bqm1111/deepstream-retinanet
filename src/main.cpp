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

	// app.detect();
	app.MOT();
	// app.detectAndMOT();
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

	// ConfigManager *config = new ConfigManager();
	// config->setContext();
	// std::shared_ptr<DSAppConfig> dsConf = std::dynamic_pointer_cast<DSAppConfig>(config->getConfig(ConfigType::DeepStreamApp));
	// dsConf->setProperty(DSAppProperty::FACE_FEATURE_CURL_ADDRESS, std::string("http://tainp.local:5555/search"));
	// dsConf->setProperty(DSAppProperty::KAFKA_TOPIC, std::string("XFace"));
	// dsConf->setProperty(DSAppProperty::KAFKA_CONNECTION_STR, std::string("localhost;9092"));
	// dsConf->setProperty(DSAppProperty::MUXER_OUTPUT_WIDTH, 1920);
	// dsConf->setProperty(DSAppProperty::MUXER_OUTPUT_HEIGHT, 1080);
	// dsConf->setProperty(DSAppProperty::TILER_COLS, 2);
	// dsConf->setProperty(DSAppProperty::TILER_ROWS, 1);
	// dsConf->setProperty(DSAppProperty::TILER_WIDTH, 1280);
	// dsConf->setProperty(DSAppProperty::TILER_HEIGHT, 480);
	// dsConf->save();
}
