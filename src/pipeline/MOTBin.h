#ifndef MOTBIN_H_764b8ce325dd4743054ac8de
#define MOTBIN_H_764b8ce325dd4743054ac8de

#include "NvInferBinBase.h"
#include "NvInferBinConfigBase.h"
#include <gst/gstelement.h>
#include "params.h"
#include "mot_struct.h"

class NvInferMOTBinConfig : public NvInferBinConfigBase
{
public:
    NvInferMOTBinConfig(std::string pgie, std::string sgie) : NvInferBinConfigBase(pgie, sgie)
    {
    }

    ~NvInferMOTBinConfig() = default;
};

class NvInferMOTBin : public NvInferBinBase
{
public:
    NvInferMOTBin(std::shared_ptr<NvInferMOTBinConfig> configs)
    {
        m_configs = configs;
        m_module_name = "mot";
    }
    ~NvInferMOTBin();
    void createInferBin() override;
    void attachProbe() override;
    void setMsgBrokerConfig() override;
    void acquireTrackerList(MOTTrackerList *tracker_list);
    static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
    static GstPadProbeReturn sgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);

private:
    MOTTrackerList *m_tracker_list;
};

#endif
