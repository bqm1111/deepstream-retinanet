#ifndef FACE_BIN_H_869d13f644651ab0365cf4a2
#define FACE_BIN_H_869d13f644651ab0365cf4a2

#include <gst/gst.h>
#include <gst/gstelement.h>
#include <gst/gstelementfactory.h>
#include <gst/gstobject.h>
#include "probe.h"
#include "params.h"

struct FaceBinConfigs
{
    const char *pgie_config_path;
    const char *aligner_config_path;
    const char *sgie_config_path;
};

struct FaceBinBackbone
{
    GstElement *pgie = NULL;
    GstElement *aligner = NULL;
    GstElement *sgie = NULL;
    GstElement *namer = NULL;
};

class FaceBin
{
public:
    FaceBin(FaceBinConfigs configs);
    void getMasterBin(GstElement *&bin);
private:
    FaceBinConfigs m_configs;
    GstElement * m_masterBin = NULL;
    FaceBinBackbone m_backbone;
    
    void createDetectBin();
    void createFullBin();
};

#endif