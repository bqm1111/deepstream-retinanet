#include <opencv2/opencv.hpp>
#include "alignment_logic.h"

static std::vector<std::string> NvBufSurfaceColorFormatPrint
{
  "NVBUF_COLOR_FORMAT_INVALID",
  "NVBUF_COLOR_FORMAT_GRAY8",
  "NVBUF_COLOR_FORMAT_YUV420",
  "NVBUF_COLOR_FORMAT_YVU420",
  "NVBUF_COLOR_FORMAT_YUV420_ER",
  "NVBUF_COLOR_FORMAT_YVU420_ER",
  "NVBUF_COLOR_FORMAT_NV12",
  "NVBUF_COLOR_FORMAT_NV12_ER",
  "NVBUF_COLOR_FORMAT_NV21",
  "NVBUF_COLOR_FORMAT_NV21_ER",
  "NVBUF_COLOR_FORMAT_UYVY",
  "NVBUF_COLOR_FORMAT_UYVY_ER",
  "NVBUF_COLOR_FORMAT_VYUY",
  "NVBUF_COLOR_FORMAT_VYUY_ER",
  "NVBUF_COLOR_FORMAT_YUYV",
  "NVBUF_COLOR_FORMAT_YUYV_ER",
  "NVBUF_COLOR_FORMAT_YVYU",
  "NVBUF_COLOR_FORMAT_YVYU_ER",
  "NVBUF_COLOR_FORMAT_YUV444",
  "NVBUF_COLOR_FORMAT_RGBA",
  "NVBUF_COLOR_FORMAT_BGRA",
  "NVBUF_COLOR_FORMAT_ARGB",
  "NVBUF_COLOR_FORMAT_ABGR",
  "NVBUF_COLOR_FORMAT_RGBx",
  "NVBUF_COLOR_FORMAT_BGRx",
  "NVBUF_COLOR_FORMAT_xRGB",
  "NVBUF_COLOR_FORMAT_xBGR",
  "NVBUF_COLOR_FORMAT_RGB",
  "NVBUF_COLOR_FORMAT_BGR",
  "NVBUF_COLOR_FORMAT_NV12_10LE",
  "NVBUF_COLOR_FORMAT_NV12_12LE",
  "NVBUF_COLOR_FORMAT_YUV420_709",
  "NVBUF_COLOR_FORMAT_YUV420_709_ER",
  "NVBUF_COLOR_FORMAT_NV12_709",
  "NVBUF_COLOR_FORMAT_NV12_709_ER",
  "NVBUF_COLOR_FORMAT_YUV420_2020",
  "NVBUF_COLOR_FORMAT_NV12_2020",
  "NVBUF_COLOR_FORMAT_NV12_10LE_ER",
  "NVBUF_COLOR_FORMAT_NV12_10LE_709",
  "NVBUF_COLOR_FORMAT_NV12_10LE_709_ER",
  "NVBUF_COLOR_FORMAT_NV12_10LE_2020",
  "NVBUF_COLOR_FORMAT_SIGNED_R16G16",
  "NVBUF_COLOR_FORMAT_R8_G8_B8",
  "NVBUF_COLOR_FORMAT_B8_G8_R8",
  "NVBUF_COLOR_FORMAT_R32F_G32F_B32F",
  "NVBUF_COLOR_FORMAT_B32F_G32F_R32F",
  "NVBUF_COLOR_FORMAT_YUV422",
  "NVBUF_COLOR_FORMAT_LAST"
};

AlignmentLogic::AlignmentLogic(const NvDsFaceAlignTensorParams &tensorParams)
{
  cudaError_t cudaReturn = cudaStreamCreate(&m_stream);
  if (cudaReturn != cudaSuccess)
  {
    g_printerr("%s:%d Failed to create cuda stream for face alignment. cudaStreamCreateWithFlags failed with error %s\n",
               __FILE__, __LINE__, cudaGetErrorName(cudaReturn));
  }

  output_c = tensorParams.network_input_shape[1];
  output_h = tensorParams.network_input_shape[2];
  output_w = tensorParams.network_input_shape[3];

  cudaReturn = cudaMalloc((void **)&gpu_trans, 3 * 3 * sizeof(float));
  if (cudaReturn != cudaSuccess)
  {
    g_printerr("%s:%d Failed to create gpu translation matrix with error %s\n", __FILE__, __LINE__, cudaGetErrorName(cudaReturn));
  }
}

AlignmentLogic::~AlignmentLogic()
{
  if (m_stream)
  {
    cudaStreamDestroy(m_stream);
    m_stream = NULL;
  }
  cudaFree(gpu_trans);
}

NvDsFaceAlignStatus AlignmentLogic::CustomTensorPreparation(std::vector<NvDsFaceAlignUnit> units, NvDsFaceAlignCustomBuf *buf)
{
  NvDsFaceAlignStatus status = NVDSFACEALIGN_TENSOR_NOT_READY;
  for(int i = 0; i < units.size(); i++) {
    status = CustomTensorPreparation(&units.at(i), buf, i);
    if (NVDSFACEALIGN_SUCCESS != status) {
      return status;
    }
  }
  return status;
}

NvDsFaceAlignStatus AlignmentLogic::CustomTensorPreparation(NvDsFaceAlignUnit *unit, NvDsFaceAlignCustomBuf *buf, const int& current_face_in_batch)
{
  if (!buf) 
    return NVDSFACEALIGN_CUSTOM_TENSOR_FAILED;
  if (!unit)
    return NVDSFACEALIGN_CUSTOM_TENSOR_FAILED;
  if (unit->is_pad)
    return NVDSFACEALIGN_SUCCESS;
  NvDsFaceAlignStatus status = NVDSFACEALIGN_TENSOR_NOT_READY;

  float *outPtr = (float *)(buf->memory_ptr) + current_face_in_batch * output_c * output_h * output_w;

  /**
   * NOTE:  do the alignment
   * input image is in unit->input_surf_params;
   * input landmark is unit->facealign_meta;
   * output is outPtr
   */
  /** NOTE: alignment here */
  cudaError_t alignment_cuda_result;
  NvBufSurfaceColorFormat bufColorFormat = unit->input_surf_params->colorFormat;
  if (bufColorFormat == NVBUF_COLOR_FORMAT_RGBA ) {
    /** access device image */
    // cv::cuda::GpuMat gpuMat(unit->input_surf_params->height,
    //                     unit->input_surf_params->width, CV_8UC4,
    //                     unit->input_surf_params->dataPtr,
    //                     unit->input_surf_params->pitch);
    // cv::Mat toSave;
    // gpuMat.download(toSave);
    // cv::imwrite("test_gpu_rgba_" + std::to_string(unit->frame_meta->batch_id) + ".jpg", toSave);

    float alignment_error;
    alignment_cuda_result = get_align_blob_RGBA(
      reinterpret_cast<unsigned char*>(unit->input_surf_params->dataPtr), 
      unit->input_surf_params->width, 
      unit->input_surf_params->height,
      unit->input_surf_params->pitch,
      unit->face_meta->faceMark,
      outPtr, 
      alignment_error
    );
  } else if (bufColorFormat == NVBUF_COLOR_FORMAT_NV12 || 
    bufColorFormat == NVBUF_COLOR_FORMAT_NV12_709 || 
    bufColorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER) {
      /** access image device NV12 memory */
    // cv::cuda::GpuMat gpuMat(unit->input_surf_params->height * 3/2,
    //                         unit->input_surf_params->width, CV_8UC1,
    //                         unit->input_surf_params->dataPtr,
    //                         unit->input_surf_params->pitch);
    // cv::Mat toSave;
    // gpuMat.download(toSave);
    // cv::imwrite("test_gpu_nv12_" + std::to_string(unit->frame_meta->batch_id) + ".jpg", toSave);

    float alignment_error;
    alignment_cuda_result = get_align_blob_NV12(
      reinterpret_cast<unsigned char*>(unit->input_surf_params->dataPtr), 
      unit->input_surf_params->width, 
      unit->input_surf_params->height,
      unit->input_surf_params->pitch,
      unit->face_meta->faceMark,
      outPtr, 
      alignment_error
    );
  } else {
    g_printerr("%s:%d: This color format (%d %s) is not supported by face alignment\n", 
      __FILE__, __LINE__, bufColorFormat, NvBufSurfaceColorFormatPrint[int(bufColorFormat)].c_str());
    return NVDSFACEALIGN_CUSTOM_TENSOR_FAILED;
  }


  if (cudaSuccess != alignment_cuda_result) {
    g_printerr("%s:%d get_align_blob_RGBA failed \n", __FILE__, __LINE__);
    status = NVDSFACEALIGN_CUDA_ERROR;
  } else {
    status = NVDSFACEALIGN_SUCCESS;
  }

  return status;
}

float AlignmentLogic::getSimilarityTransformError(float *landmark, const std::vector<cv::Point2f> &ref_points, cv::Mat &trans)
{
  cv::Mat M;
  std::vector<cv::Point2f> cur_pts(5);
  for (int i = 0; i < 5; i++)
  {
    cur_pts[i] = cv::Point2f(landmark[2 * i], landmark[2 * i + 1]);
  }

  float rmse;
  M = cv::videostab::estimateGlobalMotionRansac(ref_points, cur_pts,
                                                // cv::videostab::MM_HOMOGRAPHY,
                                                cv::videostab::MM_SIMILARITY,
                                                ransacParams,
                                                &rmse);
  M.copyTo(trans);
  return rmse;
}

cudaError_t AlignmentLogic::get_align_blob_RGBA(unsigned char *d_rgba,
                                          int d_rgba_width,
                                          int d_rgba_height,
                                          int d_rgba_pitch,
                                          float *landmark,
                                          float *d_rgbAlign,
                                          float& alignment_error)
{
  cv::Mat trans;
  float error = getSimilarityTransformError(landmark, arcface_src, trans);
  alignment_error = error;
  ALIGNMENT_CUDA_CHECK(cudaMemcpyAsync(gpu_trans, trans.data, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice, m_stream));
  return gpu_invWarpRGBAToRGBBlob(d_rgba, d_rgbAlign, gpu_trans, d_rgba_width, d_rgba_height, d_rgba_pitch,
                                  0, 0, output_w, output_h, m_stream);
}

cudaError_t AlignmentLogic::get_align_blob_NV12(unsigned char *d_rgba,
                                          int d_rgba_width,
                                          int d_rgba_height,
                                          int d_rgba_pitch,
                                          float *landmark,
                                          float *d_rgbAlign,
                                          float& alignment_error)
{
  cv::Mat trans;
  float error = getSimilarityTransformError(landmark, arcface_src, trans);
  alignment_error = error;
  ALIGNMENT_CUDA_CHECK(cudaMemcpyAsync(gpu_trans, trans.data, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice, m_stream));
  cudaError_t cudaerror;
  cudaerror = gpu_invWarpNV12ToRGBBlob(d_rgba, d_rgbAlign, gpu_trans, d_rgba_width, d_rgba_height, d_rgba_pitch,
                                  0, 0, output_w, output_h, m_stream);
  ALIGNMENT_CUDA_CHECK(cudaerror);
  return cudaerror;
}