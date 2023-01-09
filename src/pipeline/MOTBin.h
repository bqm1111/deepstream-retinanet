#ifndef MOTBIN_H_764b8ce325dd4743054ac8de
#define MOTBIN_H_764b8ce325dd4743054ac8de

#include "NvInferBinBase.h"
#include "NvInferBinConfigBase.h"
#include <gst/gstelement.h>

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
    void setConfig(std::shared_ptr<NvInferMOTBinConfig> configs);
    void createInferBin() override;
    void attachProbe() override;
    static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
    static GstPadProbeReturn sgie_src_pad_buffer_probe_VNU(GstPad *pad, GstPadProbeInfo *info, gpointer _udata);
    static GstPadProbeReturn sgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

};

#endif
