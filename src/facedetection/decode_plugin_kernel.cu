#include "decode_plugin.h"
#include "cuda_runtime_api.h"
#include "assert.h"

namespace
{
    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }
    
    template <typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

namespace nvinfer1 {

DecodePlugin::DecodePlugin()
{

}

DecodePlugin::DecodePlugin(const void *data, size_t length)
{
    const char *d = static_cast<const char*>(data);
    read(d, mInputH);
    read(d, mInputW);
}

DecodePlugin::DecodePlugin(int input_H, int input_W)
    : mInputH(input_H), mInputW(input_W)
{

}

DecodePlugin::~DecodePlugin()
{

}

void DecodePlugin::configurePlugin(const PluginTensorDesc *in, int32_t nbInput, const PluginTensorDesc *out, int32_t nbOutput) TRT_NOEXCEPT
{

}

bool DecodePlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) const TRT_NOEXCEPT
{
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

IPluginV2IOExt *DecodePlugin::clone() const TRT_NOEXCEPT
{
    DecodePlugin *p = new DecodePlugin(mInputH, mInputW);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

DataType DecodePlugin::getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const TRT_NOEXCEPT
{
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool DecodePlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool *inputIsBroadcasted, int32_t nbInputs) const TRT_NOEXCEPT
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DecodePlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const TRT_NOEXCEPT
{
    return false;
}

void DecodePlugin::attachToContext(cudnnContext *, cublasContext *, IGpuAllocator *) TRT_NOEXCEPT
{

}

void DecodePlugin::detachFromContext() TRT_NOEXCEPT
{

}

const char *DecodePlugin::getPluginType() const TRT_NOEXCEPT
{
    return "Decode_TRT";
}

const char *DecodePlugin::getPluginVersion() const TRT_NOEXCEPT
{
    return "1";
}

int32_t DecodePlugin::getNbOutputs() const TRT_NOEXCEPT
{
    return 1;
}

int32_t DecodePlugin::getOutBufferSize()
{
    int totalCount = 0;
    totalCount += mInputH /  8 * mInputW /  8 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
    totalCount += mInputH / 16 * mInputW / 16 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
    totalCount += mInputH / 32 * mInputW / 32 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);

    // The extra first element of output will be the number of detected bboxes
    // hence totalCount + 1
    totalCount += 1;
    return totalCount;
}

Dims DecodePlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) TRT_NOEXCEPT
{
    return Dims3(getOutBufferSize(), 1, 1);
}

void DecodePlugin::terminate() TRT_NOEXCEPT
{

}

size_t DecodePlugin::getWorkspaceSize(int32_t maxBatchSize) const TRT_NOEXCEPT
{
    return 0;
}

#if NV_TENSORRT_MAJOR >= 8
int32_t DecodePlugin::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT
#else
int32_t DecodePlugin::enqueue(int32_t batchSize, void const* const* inputs, void** outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
#endif
{
    forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, batchSize);
    return 0;
}

size_t DecodePlugin::getSerializationSize() const TRT_NOEXCEPT
{
    return sizeof(mInputH) + sizeof(mInputW);
}

void DecodePlugin::serialize(void *buffer) const TRT_NOEXCEPT
{
    char *d = static_cast<char*>(buffer);
    write(d, mInputH);
    write(d, mInputW);
}

void DecodePlugin::destroy() TRT_NOEXCEPT
{
    delete this;
}

void DecodePlugin::setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT
{
    mPluginNamespace = pluginNamespace;
}

const char *DecodePlugin::getPluginNamespace() const TRT_NOEXCEPT
{
    return mPluginNamespace;
}

__device__ float Logist(float data){ return 1./(1. + expf(-data)); };

__global__ void CalDetection(const float *input, float *output, int num_elem, int step, int anchor, int output_elem,
                            int netH, int netW) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_elem) return;

    int h = netH / step;
    int w = netW / step;
    int total_grid = h * w;
    int bn_idx = idx / total_grid;
    idx = idx - bn_idx * total_grid;
    int y = idx / w;
    int x = idx % w;
    const float* cur_input = input + bn_idx * (4 + 2 + 10) * 2 * total_grid;
    const float *bbox_reg = &cur_input[0];
    const float *cls_reg = &cur_input[2 * 4 * total_grid];
    const float *lmk_reg = &cur_input[2 * 4 * total_grid + 2 * 2 * total_grid];
    // 
    for (int k = 0; k < 2; ++k) {
        float conf1 = cls_reg[idx + k * total_grid * 2];
        float conf2 = cls_reg[idx + k * total_grid * 2 + total_grid];
        conf2 = expf(conf2) / (expf(conf1) + expf(conf2));
        if (conf2 <= 0.02) continue;

        float *res_count = output + bn_idx * output_elem;
        int count = (int)atomicAdd(res_count, 1);
        char* data = (char *)res_count + sizeof(float) + count * sizeof(decodeplugin::Detection);
        decodeplugin::Detection* det = (decodeplugin::Detection*)(data);

        float prior[4];
        prior[0] = ((float)x + 0.5) / w;
        prior[1] = ((float)y + 0.5) / h;
        prior[2] = (float)anchor * (k + 1) / netW;
        prior[3] = (float)anchor * (k + 1) / netH;

        // Location
        det->bbox[0] = prior[0] + bbox_reg[idx + k * total_grid * 4] * 0.1 * prior[2];
        det->bbox[1] = prior[1] + bbox_reg[idx + k * total_grid * 4 + total_grid] * 0.1 * prior[3];
        det->bbox[2] = prior[2] * expf(bbox_reg[idx + k * total_grid * 4 + total_grid * 2] * 0.2);
        det->bbox[3] = prior[3] * expf(bbox_reg[idx + k * total_grid * 4 + total_grid * 3] * 0.2);
        det->bbox[0] -= det->bbox[2] / 2;
        det->bbox[1] -= det->bbox[3] / 2;
        det->bbox[2] += det->bbox[0];
        det->bbox[3] += det->bbox[1];
        det->bbox[0] *= netW;
        det->bbox[1] *= netH;
        det->bbox[2] *= netW;
        det->bbox[3] *= netH;
        det->class_confidence = conf2;
        for (int i = 0; i < 10; i += 2) {
            det->landmark[i] = prior[0] + lmk_reg[idx + k * total_grid * 10 + total_grid * i] * 0.1 * prior[2];
            det->landmark[i+1] = prior[1] + lmk_reg[idx + k * total_grid * 10 + total_grid * (i + 1)] * 0.1 * prior[3];
            det->landmark[i] *= netW;
            det->landmark[i+1] *= netH;
        }
    }
}


void DecodePlugin::forwardGpu(const float * const *inputs, float *output, cudaStream_t stream, int batchSize)
{
    int num_elem = 0;
    int base_step = 8;
    int base_anchor = 16;
    int thread_count;

    int totalCount = getOutBufferSize();
    for(int idx = 0 ; idx < batchSize; ++idx) {
        cudaMemset(output + idx * totalCount, 0, sizeof(float));
    }

    for (unsigned int i = 0; i < 3; ++i)
    {
        num_elem = batchSize * mInputH / base_step * mInputW / base_step;
        thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
        CalDetection<<< (num_elem + thread_count - 1) / thread_count, thread_count, 0, stream>>>
            (inputs[i], output, num_elem, base_step, base_anchor, totalCount, mInputH, mInputW);
        base_step *= 2;
        base_anchor *= 4;
    }
}

int DecodePlugin::initialize() TRT_NOEXCEPT
{
    return 0;
}


PluginFieldCollection DecodePluginCreator::mFC {};
std::vector<PluginField> DecodePluginCreator::mPluginAttributes;

DecodePluginCreator::DecodePluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

DecodePluginCreator::~DecodePluginCreator()
{

}

const char *DecodePluginCreator::getPluginName() const TRT_NOEXCEPT
{
    return "Decode_TRT";
}

const char *DecodePluginCreator::getPluginVersion() const TRT_NOEXCEPT
{
    return "1";
}

const PluginFieldCollection *DecodePluginCreator::getFieldNames() TRT_NOEXCEPT
{
    return &mFC;
}

IPluginV2 *DecodePluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT
{
    assert (fc->nbFields == 2);
    const PluginField field_H = (fc->fields)[0];
    const PluginField field_W = (fc->fields)[1];
    const int* _temp_H = (const int*)(field_H.data);
    const int* _temp_W = (const int*)(field_W.data);
    int inputH = _temp_H[0];
    int inputW = _temp_W[0];
    DecodePlugin* obj = new DecodePlugin(inputH, inputW);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2 *DecodePluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT
{
    // This object will be deleted when the network is destroyed, which will
    // call DecodePlugin::destroy()
    DecodePlugin* obj = new DecodePlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void DecodePluginCreator::setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT
{
    mNamespace = pluginNamespace;
}

const char *DecodePluginCreator::getPluginNamespace() const TRT_NOEXCEPT
{
    return mNamespace.c_str();
}
}
