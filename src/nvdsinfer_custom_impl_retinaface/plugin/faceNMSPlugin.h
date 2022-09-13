#ifndef FACE_NMS_PLUGIN_H
#define FACE_NMS_PLUGIN_H
// #include "common/kernel.h"
// #include "common/nmsUtils.h"
// #include "common/plugin.h"
// #include "gatherNMSOutputs.h"
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include <NvInferRuntime.h>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include "plugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
    namespace plugin
    {
        struct NMSParams
        {
            NMSParams(float iou_thresh = 0.4, float post_cluster_thresh = 0.7, int tK = 300)
            {
                iou_threshold = iou_thresh;
                post_cluster_threshold = post_cluster_thresh;
                topK = tK;
            }
            float iou_threshold;
            float post_cluster_threshold;
            int topK;
        };
        class FaceNMSDynamicPlugin : public IPluginV2DynamicExt
        {
        public:
            FaceNMSDynamicPlugin(NMSParams param);
            FaceNMSDynamicPlugin(const void *data, size_t length);
            ~FaceNMSDynamicPlugin() override = default;

            // IPluginV2 methods
            const char *getPluginType() const noexcept override;
            const char *getPluginVersion() const noexcept override;
            int getNbOutputs() const noexcept override;
            int initialize() noexcept override;
            void terminate() noexcept override;
            size_t getSerializationSize() const noexcept override;
            void serialize(void *buffer) const noexcept override;
            void destroy() noexcept override;
            void setPluginNamespace(const char *libNamespace) noexcept override;
            const char *getPluginNamespace() const noexcept override;
            void setClipParam(bool clip) noexcept;
            void setScoreBits(int32_t scoreBits) noexcept;
            void setCaffeSemantics(bool caffeSemantics) noexcept;

            // IPluginV2Ext methods
            nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputType, int nbInputs) const
                noexcept override;

            // IPluginV2DynamicExt methods
            IPluginV2DynamicExt *clone() const noexcept override;
            DimsExprs getOutputDimensions(
                int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) noexcept override;
            bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept override;
            void configurePlugin(
                const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
            size_t getWorkspaceSize(
                const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const noexcept override;
            int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs,
                        void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

        private:
            NMSParams param{};
            
            // int boxesSize{};
            // int scoresSize{};
            // int numPriors{};
            // bool mClipBoxes{};
            // DataType mPrecision;
            // int32_t mScoreBits;
            // bool mCaffeSemantics{true};
            std::string mNamespace;
            pluginStatus_t mPluginStatus{};
        };

        class FaceNMSBasePluginCreator : public nvinfer1::pluginInternal::BaseCreator
        {
        public:
            FaceNMSBasePluginCreator();
            ~FaceNMSBasePluginCreator() override = default;

            const char *getPluginVersion() const noexcept override;
            const PluginFieldCollection *getFieldNames() noexcept override;

        protected:
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
        };

        class FaceNMSDynamicPluginCreator : public FaceNMSBasePluginCreator
        {
        public:
            const char *getPluginName() const noexcept override;
            IPluginV2DynamicExt *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
            IPluginV2DynamicExt *deserializePlugin(
                const char *name, const void *serialData, size_t serialLength) noexcept override;
        };

    }
}

#endif