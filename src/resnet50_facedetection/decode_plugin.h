#ifndef RETINAFACE_PLUGIN_H
#define RETINAFACE_PLUGIN_H

#include <string>
#include <vector>
#include "NvInfer.h"

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#else 
#define TRT_NOEXCEPT
#endif // NV_TENSORRT_MAJOR >= 8

namespace decodeplugin {
	struct alignas(float) Detection {
		float bbox[4]; // x1 y1 x2 y2
		float class_confidence;
		float landmark[10];
	};
	static const char* inputH = "inputH";
	static const char* inputW = "inputW";
}

namespace nvinfer1 {

class DecodePlugin: public IPluginV2IOExt
{
public:
	DecodePlugin();
	DecodePlugin(const void* data, size_t length);
	DecodePlugin(int input_H, int input_W);

	~DecodePlugin();

	// override all pure virtual method of IPluginV2IOExt
	virtual void configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out, int32_t nbOutput) TRT_NOEXCEPT override;
	virtual bool supportsFormatCombination(
			int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) const TRT_NOEXCEPT override;
	virtual IPluginV2IOExt* clone() const TRT_NOEXCEPT override;

	// override all pure virtual method of IPluginV2Ext
	virtual nvinfer1::DataType getOutputDataType(
		int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const TRT_NOEXCEPT override;
	virtual bool isOutputBroadcastAcrossBatch(
		int32_t outputIndex, const bool* inputIsBroadcasted, int32_t nbInputs) const TRT_NOEXCEPT override;
	virtual bool canBroadcastInputAcrossBatch(int32_t inputIndex) const TRT_NOEXCEPT override;

	// deo hieu sao phai override these
	virtual void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) TRT_NOEXCEPT override;
	virtual void detachFromContext() TRT_NOEXCEPT override;

	// override all pure virtual method of IPluginV2
	const char* getPluginType() const TRT_NOEXCEPT override;
	const char* getPluginVersion() const TRT_NOEXCEPT override;
	virtual int32_t getNbOutputs() const TRT_NOEXCEPT override;
	virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;
	virtual int32_t initialize() TRT_NOEXCEPT override;
	virtual void terminate() TRT_NOEXCEPT override;
	virtual size_t getWorkspaceSize(int32_t maxBatchSize) const TRT_NOEXCEPT override;
#if NV_TENSORRT_MAJOR >= 8
	virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
		cudaStream_t stream) TRT_NOEXCEPT override;
#else
	virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void** outputs, void* workspace,
		cudaStream_t stream) TRT_NOEXCEPT override;
#endif
	virtual size_t getSerializationSize() const TRT_NOEXCEPT override;
	virtual void serialize(void* buffer) const TRT_NOEXCEPT override;
	virtual void destroy() TRT_NOEXCEPT override;
	virtual void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;
	virtual const char* getPluginNamespace() const TRT_NOEXCEPT override;

	int32_t getOutBufferSize();
private:
	void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);
	int thread_count_ = 512;
	const char* mPluginNamespace;

	int mInputH;
	int mInputW;
};
// 
class DecodePluginCreator : public IPluginCreator
{
public:
	DecodePluginCreator();
	~DecodePluginCreator() override;

	virtual const char* getPluginName() const TRT_NOEXCEPT override;
	virtual const char* getPluginVersion() const TRT_NOEXCEPT override;
	virtual const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;
	virtual IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;
	virtual IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override;
	virtual void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;
	virtual const char* getPluginNamespace() const TRT_NOEXCEPT override;
private:
	std::string mNamespace;
	static PluginFieldCollection mFC;
	static std::vector<PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);

} // namespace nvinfer1

#endif // RETINAFACE_PLUGIN_H
