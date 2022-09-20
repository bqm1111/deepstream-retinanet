#include "faceApp.h"
#include "DeepStreamAppConfig.h"

FaceApp::FaceApp()
{
    m_config = new ConfigManager();
    init_curl();
}

FaceApp::~FaceApp()
{
    delete m_config;
    free_curl();
    // printf("tracker list = %p\n", m_tracker_list->trackers);
    // if (m_tracker_list->trackers != NULL)
    // {
    //     g_free(m_tracker_list->trackers);
    // }

    if (!m_tracker_list)
    {
        g_free(m_tracker_list);
    }
}

static void printParam(GstAppParam param)
{
    std::cout << param.muxer_output_height << std::endl
              << param.muxer_output_width
              << param.tiler_cols << std::endl
              << param.tiler_rows << std::endl
              << param.tiler_width << std::endl
              << param.tiler_height << std::endl
              << param.face_topic << std::endl
              << param.msg_config_path << std::endl
              << param.connection_str << std::endl
              << param.msg2p_lib << std::endl
              << param.proto_lib << std::endl;
}

void FaceApp::loadConfig(std::string config_file)
{
    m_config->setContext();
    std::shared_ptr<DSAppConfig> appConf = std::dynamic_pointer_cast<DSAppConfig>(m_config->getConfig(ConfigType::DeepStreamApp));

    m_gstparam.muxer_output_height = appConf->getProperty(DSAppProperty::MUXER_OUTPUT_HEIGHT).toInt();
    m_gstparam.muxer_output_width = appConf->getProperty(DSAppProperty::MUXER_OUTPUT_WIDTH).toInt();
    m_gstparam.tiler_cols = appConf->getProperty(DSAppProperty::TILER_COLS).toInt();
    m_gstparam.tiler_rows = appConf->getProperty(DSAppProperty::TILER_ROWS).toInt();
    m_gstparam.tiler_width = appConf->getProperty(DSAppProperty::TILER_WIDTH).toInt();
    m_gstparam.tiler_height = appConf->getProperty(DSAppProperty::TILER_HEIGHT).toInt();

    m_gstparam.face_topic = appConf->getProperty(DSAppProperty::FACE_KAFKA_TOPIC).toString();
    m_gstparam.mot_topic = appConf->getProperty(DSAppProperty::MOT_KAFKA_TOPIC).toString();

    m_gstparam.msg_config_path = appConf->getProperty(DSAppProperty::MSG_CONFIG_PATH).toString();
    m_gstparam.connection_str = appConf->getProperty(DSAppProperty::KAFKA_CONNECTION_STR).toString();
    m_gstparam.msg2p_lib = appConf->getProperty(DSAppProperty::KAFKA_MSG2P_LIB).toString();
    m_gstparam.proto_lib = appConf->getProperty(DSAppProperty::KAFKA_PROTO_LIB).toString();
}

void FaceApp::create(std::string name)
{
    m_pipeline.create(name);
}

void FaceApp::addVideoSource(std::string list_video_src_file)
{
    parseJson(list_video_src_file, m_video_source_name, m_video_source_info);
    m_pipeline.add_video_source(m_video_source_info, m_video_source_name);
}

void FaceApp::init_curl()
{
    m_curl = curl_easy_init();
    assert(m_curl);

    /* copy from postman */
    curl_easy_setopt(m_curl, CURLOPT_CUSTOMREQUEST, "POST");
    curl_easy_setopt(m_curl, CURLOPT_URL, "http://tainp.local:5555/search");

    // curl_easy_setopt(m_curl, CURLOPT_VERBOSE, 11);

    /* HTTP/2 */
    curl_easy_setopt(m_curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);

    /* No SSL */
    curl_easy_setopt(m_curl, CURLOPT_SSL_VERIFYPEER, 0);

    /* wait for pipe connection to confirm*/
    curl_easy_setopt(m_curl, CURLOPT_PIPEWAIT, 1L);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(m_curl, CURLOPT_HTTPHEADER, headers);
}

void FaceApp::free_curl()
{
    curl_easy_cleanup(m_curl);
}
GstElement *FaceApp::getPipeline()
{
    return m_pipeline.m_pipeline;
}

int FaceApp::numVideoSrc()
{
    return m_video_source_name.size();
}

void FaceApp::faceDetection()
{
    // std::shared_ptr<NvInferFaceBinConfig> face_configs(std::make_shared<NvInferFaceBinConfig>(FACEID_PGIE_CONFIG_PATH, FACEID_SGIE_CONFIG_PATH, FACEID_ALIGN_CONFIG_PATH));
    // NvInferFaceBin face_bin(face_configs);
    // m_pipeline.linkMuxer();

    // m_pipeline.attachOsdProbe(face_bin.osd_sink_pad_callback);
    // m_pipeline.attachTileProbe(face_bin.tiler_sink_pad_buffer_probe);
    // // m_app.createFileSinkBin("out.mp4");

    // GstElement *bin = NULL;
    // face_bin.getMasterBin(bin);
    // gst_bin_add(GST_BIN(m_pipeline.m_pipeline), bin);
    // if (!gst_element_link_many(m_pipeline.m_stream_muxer, bin, m_pipeline.m_tiler, NULL))
    // {
    //     g_printerr("%s:%d Cant link face detection bin\n", __FILE__, __LINE__);
    // }

    // GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pipeline.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}

void FaceApp::detectAndSend()
{
    std::shared_ptr<NvInferFaceBinConfig> face_configs = std::make_shared<NvInferFaceBinConfig>(FACEID_PGIE_CONFIG_PATH, FACEID_SGIE_CONFIG_PATH, FACEID_ALIGN_CONFIG_PATH);
    NvInferFaceBin face_bin(face_configs);
    // remember to acquire curl before createBin
    face_bin.setParam(m_gstparam);

    face_bin.acquireCurl(m_curl);
    m_pipeline.linkMuxer(m_gstparam.muxer_output_width, m_gstparam.muxer_output_height);
    GstElement *inferbin = face_bin.createInferPipeline(m_pipeline.m_pipeline);
    face_bin.attachProbe();
    face_bin.setMsgBrokerConfig();

    if (!gst_element_link_many(m_pipeline.m_stream_muxer, inferbin, NULL))
    {
        g_printerr("%s:%d Cant link inferbin to detect and Send\n", __FILE__, __LINE__);
    }
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pipeline.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}

void FaceApp::MOT()
{
    m_tracker_list = (MOTTrackerList *)g_malloc0(sizeof(MOTTrackerList));
    int num_tracker = m_video_source_info.size();
    m_tracker_list->trackers = (tracker *)g_malloc0(sizeof(tracker) * num_tracker);
    m_tracker_list->num_trackers = num_tracker;
    for (size_t i = 0; i < m_tracker_list->num_trackers; i++)
        this->m_tracker_list->trackers[i] = tracker(
            0.1363697015033318, 91, 0.7510890862625559, 18, 2, 1.);

    std::shared_ptr<NvInferMOTBinConfig> mot_configs = std::make_shared<NvInferMOTBinConfig>(MOT_PGIE_CONFIG_PATH, MOT_SGIE_CONFIG_PATH);
    NvInferMOTBin mot_bin(mot_configs);
    // remember to acquire trackerList before createBin

    mot_bin.acquireTrackerList(m_tracker_list);
    mot_bin.setParam(m_gstparam);
    m_pipeline.linkMuxer(m_gstparam.muxer_output_width, m_gstparam.muxer_output_height);

    mot_bin.createInferBin();

    GstElement *inferbin = mot_bin.createInferPipeline(m_pipeline.m_pipeline);
    mot_bin.attachProbe();
    mot_bin.setMsgBrokerConfig();

    if (!gst_element_link_many(m_pipeline.m_stream_muxer, inferbin, NULL))
    {
        g_printerr("%s:%d Cant link mot bin\n", __FILE__, __LINE__);
    }
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(m_pipeline.m_pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "test_run");
}