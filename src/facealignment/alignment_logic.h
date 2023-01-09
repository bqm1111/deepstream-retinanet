#ifndef _ALIGNMENT_LIB_H_
#define _ALIGNMENT_LIB_H_

#include "cuda_runtime.h"
#include "nvdsfacealign_interface.h"

#include <opencv2/videostab.hpp>

#ifndef ALIGNMENT_CUDA_CHECK
#define ALIGNMENT_CUDA_CHECK(ans)                          \
    {                                                      \
        gpuAssert_BkgcPSEShjn7((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert_BkgcPSEShjn7(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUAssert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        fflush(stderr);
        if (abort)
            exit(code);
    }
}
#endif

cudaError_t gpu_invWarpNV12ToRGBBlob(const unsigned char *d_nv12, float *d_rgbBlob, const float *d_T, const int _width, const int _height, const int d_step,
                                     const int _cropX, const int _cropY, const int _cropWidth, const int _cropHeight,
                                     cudaStream_t stream);


cudaError_t gpu_invWarpRGBAToRGBBlob(const unsigned char *d_nv12, float *d_rgbBlob, const float *d_T, const int d_width, const int d_height, const int d_step,
                                     const int _cropX, const int _cropY, const int _cropWidth, const int _cropHeight,
                                     cudaStream_t stream);

/**
 * @brief Just for debug
 * 
 * @param d_rgbBlob 
 * @param size 
 * @param value 
 * @return cudaError_t 
 */
cudaError_t gpu_malloc(float* d_rgbBlob, int size, float value, cudaStream_t stream);


/**
 * @brief Where the alignment really happen
 * 
 */
class AlignmentLogic
{
public:
    AlignmentLogic(const NvDsFaceAlignTensorParams& tensorParams);
    ~AlignmentLogic();

    /**
     * @brief prepare all unit in units. output to buf
     * 
     * @param units 
     * @param buf the output
     * @return NvDsFaceAlignStatus 
     */
    NvDsFaceAlignStatus CustomTensorPreparation(std::vector<NvDsFaceAlignUnit> units, NvDsFaceAlignCustomBuf *buf);
    NvDsFaceAlignStatus CustomTensorPreparation(NvDsFaceAlignUnit *unit, NvDsFaceAlignCustomBuf *buf, const int& current_face_in_batch);

private:

    cudaError_t get_align_blob_NV12(unsigned char *d_rgba,
                              int d_rgba_width,
                              int d_rgba_height,
                              int d_rgba_pitch,
                              float *landmark,
                              float *d_rgbAlign,
                              float& alignment_error);

    cudaError_t get_align_blob_RGBA(unsigned char *d_rgba,
                              int d_rgba_width,
                              int d_rgba_height,
                              int d_rgba_pitch,
                              float *landmark,
                              float *d_rgbAlign,
                              float& alignment_error);

    /**
     * @brief Get the Similarity Transform Error object
     * 
     * @param landmark array of 10 floats representing x1 y1 x2 y2 ... x5 y5
     * @param ref_points 
     * @param trans 
     * @return float 
     */
    float getSimilarityTransformError(float *landmark, const std::vector<cv::Point2f> &ref_points, cv::Mat &trans);


    int output_c;
    int output_h;
    int output_w;
    cudaStream_t m_stream;

    float *gpu_trans;
    const std::vector<cv::Point2f> arcface_src{
        cv::Point2f(38.2946, 51.6963),
        cv::Point2f(73.5318, 51.5014),
        cv::Point2f(56.0252, 71.7366),
        cv::Point2f(41.5493, 92.3655),
        cv::Point2f(70.7299, 92.2041),
    };
    const cv::videostab::RansacParams ransacParams =
        // cv::videostab::RansacParams(4, 2.0f, 0.5f, 0.99f);
        cv::videostab::RansacParams(3, 2.0f, 0.5f, 0.99f);
};

#endif // _ALIGNMENT_LIB_H_