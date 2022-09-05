#include "retinaface.h"
#include "trt_utils.h"

RetinaFace::RetinaFace(int net_H, int net_W, int32_t maxBatchSize, nvinfer1::DataType inputType, std::string weighFile)
    : m_input_h(net_H), m_input_w(net_W), m_maxBatchSize(maxBatchSize), m_inputType(inputType), m_weight_file(weighFile)
{
    if (net_H % 32 != 0 || net_W % 32 != 0)
    {
        std::cerr << "Retinaface input size must devided to 32. You have " << net_W << "x" << net_H << std::endl;
        throw std::runtime_error("Retinaface input size must devided to 32");
    }
    m_weightMap = loadWeights(m_weight_file);
}

RetinaFace::~RetinaFace()
{
    freeWeights(m_weightMap);
}

NvDsInferStatus RetinaFace::parseModel(nvinfer1::INetworkDefinition &network)
{
    const char *INPUT_BLOB_NAME = "data";
    const char *OUTPUT_BLOB_NAME = "prob";

    std::map<std::string, nvinfer1::Weights> weightMap = m_weightMap;

    // Create input tensor with name INPUT_BLOB_NAME
    ITensor *data = network.addInput(INPUT_BLOB_NAME, m_inputType, Dims3{3, m_input_h, m_input_w});
    assert(data);

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------------- backbone resnet50 ---------------
    IConvolutionLayer *conv1 = network.addConvolutionNd(*data, 64, DimsHW{7, 7}, getWeights(weightMap, "body.conv1.weight"), emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer *bn1 = addBatchNorm2d(&network, weightMap, *conv1->getOutput(0), "body.bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer *relu1 = network.addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer *pool1 = network.addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer *x = bottleneck(&network, weightMap, *pool1->getOutput(0), 64, 64, 1, "body.layer1.0.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.1.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 256, 64, 1, "body.layer1.2.");

    x = bottleneck(&network, weightMap, *x->getOutput(0), 256, 128, 2, "body.layer2.0.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.1.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.2.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 512, 128, 1, "body.layer2.3.");
    IActivationLayer *layer2 = x;

    x = bottleneck(&network, weightMap, *x->getOutput(0), 512, 256, 2, "body.layer3.0.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.1.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.2.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.3.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.4.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 1024, 256, 1, "body.layer3.5.");
    IActivationLayer *layer3 = x;

    x = bottleneck(&network, weightMap, *x->getOutput(0), 1024, 512, 2, "body.layer4.0.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.1.");
    x = bottleneck(&network, weightMap, *x->getOutput(0), 2048, 512, 1, "body.layer4.2.");
    IActivationLayer *layer4 = x;

    // ------------- FPN ---------------
    auto output1 = conv_bn_relu(&network, weightMap, *layer2->getOutput(0), 256, 1, 1, 0, true, "fpn.output1");
    auto output2 = conv_bn_relu(&network, weightMap, *layer3->getOutput(0), 256, 1, 1, 0, true, "fpn.output2");
    auto output3 = conv_bn_relu(&network, weightMap, *layer4->getOutput(0), 256, 1, 1, 0, true, "fpn.output3");

    float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++)
    {
        deval[i] = 1.0;
    }
    Weights deconvwts{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer *up3 = network.addDeconvolutionNd(*output3->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up3);
    up3->setStrideNd(DimsHW{2, 2});
    up3->setNbGroups(256);
    weightMap["up3"] = deconvwts;

    output2 = network.addElementWise(*output2->getOutput(0), *up3->getOutput(0), ElementWiseOperation::kSUM);
    output2 = conv_bn_relu(&network, weightMap, *output2->getOutput(0), 256, 3, 1, 1, true, "fpn.merge2");

    IDeconvolutionLayer *up2 = network.addDeconvolutionNd(*output2->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
    assert(up2);
    up2->setStrideNd(DimsHW{2, 2});
    up2->setNbGroups(256);
    output1 = network.addElementWise(*output1->getOutput(0), *up2->getOutput(0), ElementWiseOperation::kSUM);
    output1 = conv_bn_relu(&network, weightMap, *output1->getOutput(0), 256, 3, 1, 1, true, "fpn.merge1");

    // ------------- SSH ---------------
    auto ssh1 = ssh(&network, weightMap, *output1->getOutput(0), "ssh1");
    auto ssh2 = ssh(&network, weightMap, *output2->getOutput(0), "ssh2");
    auto ssh3 = ssh(&network, weightMap, *output3->getOutput(0), "ssh3");

    // ------------- Head ---------------
    auto bbox_head1 = network.addConvolutionNd(*ssh1->getOutput(0), 2 * 4, DimsHW{1, 1}, getWeights(weightMap, "BboxHead.0.conv1x1.weight"), getWeights(weightMap, "BboxHead.0.conv1x1.bias"));
    auto bbox_head2 = network.addConvolutionNd(*ssh2->getOutput(0), 2 * 4, DimsHW{1, 1}, getWeights(weightMap, "BboxHead.1.conv1x1.weight"), getWeights(weightMap, "BboxHead.1.conv1x1.bias"));
    auto bbox_head3 = network.addConvolutionNd(*ssh3->getOutput(0), 2 * 4, DimsHW{1, 1}, getWeights(weightMap, "BboxHead.2.conv1x1.weight"), getWeights(weightMap, "BboxHead.2.conv1x1.bias"));

    auto cls_head1 = network.addConvolutionNd(*ssh1->getOutput(0), 2 * 2, DimsHW{1, 1}, getWeights(weightMap, "ClassHead.0.conv1x1.weight"), getWeights(weightMap, "ClassHead.0.conv1x1.bias"));
    auto cls_head2 = network.addConvolutionNd(*ssh2->getOutput(0), 2 * 2, DimsHW{1, 1}, getWeights(weightMap, "ClassHead.1.conv1x1.weight"), getWeights(weightMap, "ClassHead.1.conv1x1.bias"));
    auto cls_head3 = network.addConvolutionNd(*ssh3->getOutput(0), 2 * 2, DimsHW{1, 1}, getWeights(weightMap, "ClassHead.2.conv1x1.weight"), getWeights(weightMap, "ClassHead.2.conv1x1.bias"));

    auto lmk_head1 = network.addConvolutionNd(*ssh1->getOutput(0), 2 * 10, DimsHW{1, 1}, getWeights(weightMap, "LandmarkHead.0.conv1x1.weight"), getWeights(weightMap, "LandmarkHead.0.conv1x1.bias"));
    auto lmk_head2 = network.addConvolutionNd(*ssh2->getOutput(0), 2 * 10, DimsHW{1, 1}, getWeights(weightMap, "LandmarkHead.1.conv1x1.weight"), getWeights(weightMap, "LandmarkHead.1.conv1x1.bias"));
    auto lmk_head3 = network.addConvolutionNd(*ssh3->getOutput(0), 2 * 10, DimsHW{1, 1}, getWeights(weightMap, "LandmarkHead.2.conv1x1.weight"), getWeights(weightMap, "LandmarkHead.2.conv1x1.bias"));

    // ------------- Decode bbox, conf, landmark ---------------
    ITensor *inputTensors1[] = {bbox_head1->getOutput(0), cls_head1->getOutput(0), lmk_head1->getOutput(0)};
    auto cat1 = network.addConcatenation(inputTensors1, 3);
    ITensor *inputTensors2[] = {bbox_head2->getOutput(0), cls_head2->getOutput(0), lmk_head2->getOutput(0)};
    auto cat2 = network.addConcatenation(inputTensors2, 3);
    ITensor *inputTensors3[] = {bbox_head3->getOutput(0), cls_head3->getOutput(0), lmk_head3->getOutput(0)};
    auto cat3 = network.addConcatenation(inputTensors3, 3);

    auto creator = getPluginRegistry()->getPluginCreator("Decode_TRT", "1");
    PluginFieldCollection pfc;
    pfc.nbFields = 2;
    PluginField *fields = (PluginField *)malloc(pfc.nbFields * sizeof(PluginField));
    fields[0].name = decodeplugin::inputH;
    fields[0].data = &m_input_h;
    fields[0].type = PluginFieldType::kINT32;
    fields[0].length = 1;
    fields[1].name = decodeplugin::inputW;
    fields[1].data = &m_input_w;
    fields[1].type = PluginFieldType::kINT32;
    fields[1].length = 1;
    pfc.fields = (const PluginField *)fields;
    //
    IPluginV2 *pluginObj = creator->createPlugin("decode", &pfc);
    ITensor *inputTensors[] = {cat1->getOutput(0), cat2->getOutput(0), cat3->getOutput(0)};
    auto decodelayer = network.addPluginV2(inputTensors, 3, *pluginObj);
    assert(decodelayer);

    decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network.markOutput(*decodelayer->getOutput(0));
    return NVDSINFER_SUCCESS;
}
extern "C" bool NvDsInferRetinafaceCudaEngineGet(nvinfer1::IBuilder *const builder,
                                                 nvinfer1::IBuilderConfig *const builderConfig,
                                                 const NvDsInferContextInitParams *const initParams,
                                                 nvinfer1::DataType dataType,
                                                 nvinfer1::ICudaEngine *&cudaEngine)
{
    if (!builder)
        throw std::runtime_error("NvDsInferRetinafaceCudaEngineGet: builder is NULL\n");
    if (!builderConfig)
        throw std::runtime_error("NvDsInferRetinafaceCudaEngineGet: builderConfig is NULL\n");
    if (!initParams)
        throw std::runtime_error("NvDsInferRetinafaceCudaEngineGet: initParams is NULL\n");

    if (initParams->netInputOrder != NvDsInferTensorOrder_kNCHW ||
        initParams->networkInputFormat != NvDsInferFormat_BGR)
    {
        throw std::runtime_error("NvDsInferRetinafaceCudaEngineGet not supported netInputOrder\n");
    }

    // FIXME: pass weight from config file
    int net_H = initParams->inferInputDims.h;
    int net_W = initParams->inferInputDims.w;
    std::string weightFile = RETINAFACE_WEIGHT_PATH;
    RetinaFace retinafaceDS(net_H, net_W, builder->getMaxBatchSize(), nvinfer1::DataType::kFLOAT, weightFile);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    retinafaceDS.parseModel(*network);
    cudaEngine = builder->buildEngineWithConfig(*network, *builderConfig);
    delete network;
    if (cudaEngine != nullptr)
    {
        std::cout << "Build engine successfully!" << std::endl;
        return true;
    }
    else
    {
        std::cerr << "[ERROR] Build engine failed!" << std::endl;
        return false;
    }
}

#include <cstring>
extern "C" bool NvDsInferParseCustomRetinaface(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    // we only have one output layer
    if (outputLayersInfo.size() != 0)
    {
        throw std::runtime_error("outputLayersInfo has more than one layer");
    }

    NvDsInferLayerInfo outputLayerInfo = outputLayersInfo.at(0);
    // the output is held in outputLayerInfo.buffer
    float *output = (float *)outputLayerInfo.buffer;
    float numBboxes = output[0];

    for (int i = 0; i < numBboxes; i++)
    {
        if (output[15 * i + 1 + 4] <= detectionParams.perClassPreclusterThreshold[0])
            continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
        NvDsInferParseObjectInfo aBox;
        aBox.classId = 0; // only detect face
        aBox.left = det.bbox[0];
        aBox.top = det.bbox[1];
        aBox.width = det.bbox[2] - det.bbox[0];
        aBox.height = det.bbox[3] - det.bbox[1];
        aBox.detectionConfidence = det.class_confidence;
        objectList.push_back(aBox);
    }

    return true;
}
