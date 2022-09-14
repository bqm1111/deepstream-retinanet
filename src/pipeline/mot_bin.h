#ifndef MOT_BIN_H_6539df5b49e099728cd1bf7e
#define MOT_BIN_H_6539df5b49e099728cd1bf7e
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <cstring>

#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstobject.h>
#include "tracker.h"

struct MOTBinConfigs
{
    size_t num_trackers = 0; // must match batch-size of nvstreammux

    const char *pgie_config_path;
    const char *sgie_config_path;
};

struct MOTBinBackbone
{
    GstElement *pgie = NULL;
    GstElement *sgie = NULL;
};

struct MOTTrackerList
{
    tracker *trackers = NULL;
    size_t num_trackers;
};

class MOTBin
{
public:
    MOTBin(MOTBinConfigs configs);
    void getMasterBin(GstElement *&bin);

private:
    MOTTrackerList *m_tracker_list;
    MOTBinConfigs m_configs;
    GstElement *m_masterBin = NULL;
    MOTBinBackbone m_backbone;

    void createBin();
};

#endif