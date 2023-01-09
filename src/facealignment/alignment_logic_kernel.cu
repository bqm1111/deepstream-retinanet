#include "alignment_logic.h"
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef THREAD_PER_BLOCK
#define THREAD_PER_BLOCK 512
#endif

__constant__ __device__ float BT470[9] {
    1, 0 , 1.13983,
    1, -0.39465, -0.58060,
    1, 2.03211, 0
};

__constant__ __device__ float BT709[9] {
    1, 0 , 1.28033,
    1, -0.21482, -0.38059,
    1, 2.12798, 0
};

__device__
bool isInside(int col, int row, int width, int height)
{
	if (col < 0) return false;
	if (row < 0) return false;
	if (col >= width)  return false;
	if (row >= height) return false;
	return true;
}

/*=================================================
 *
 *  NV12 Image Affine Warping
 * @param: _color_T is either BT470 or BT709
 *===============================================*/
__global__ void cuda_invWarpNV12ToRGBBlob( const unsigned char *d_nv12, float *d_rgbBlob, const float *d_T, const int _width, const int _height, const int _step,
    const int _cropX, const int _cropY, const int _cropWidth, const int _cropHeight, float* _color_T)
{
    int len = _width * _height,
    cropLen = _cropWidth * _cropHeight;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < cropLen )
    {
        // Compute index of pixel in d_nv12 that is corresponding to current pixel in d_rgbBlob
        int row  = procIdx / _cropWidth + _cropY,
            col  = procIdx % _cropWidth + _cropX;
        float x  = (float)col + 0.5,
              y  = (float)row + 0.5;
        float invX = d_T[0] * x + d_T[1] * y + d_T[2],
              invY = d_T[3] * x + d_T[4] * y + d_T[5],
              invW = d_T[6] * x + d_T[7] * y + d_T[8];
        if( invW )
        {
            invX /= invW;
            invY /= invW;
        }

        // Compute index of 4 nearest neighbours of th imaginary source pixel
        int row0 = (invY >= 0) ? (int)invY : ((int)invY - 1),
            col0 = (invX >= 0) ? (int)invX : ((int)invX - 1);

        float dx = invX - (float)col0 - 0.5,
              dy = invY - (float)row0 - 0.5;
        int dR   = (dy < 0) ? -1 : 1,
            dC   = (dx < 0) ? -1 : 1;

        int row1 = row0,
            col1 = col0 + dC,
            row2 = row0 + dR,
            col2 = col0 + dC,
            row3 = row0 + dR,
            col3 = col0;

        // Compute bilinear weights
        float alphaX = 1.0 - fabsf(dx),
              alphaY = 1.0 - fabsf(dy);
        float betaX  = 1.0 - alphaX,
              betaY  = 1.0 - alphaY;
        float w0 = alphaX * alphaY,
              w1 = betaX * alphaY,
              w2 = betaX * betaY,
              w3 = alphaX * betaY;

        // Perform bilinear interpolation
        float yBi = 0.0, uBi = 0.0, vBi = 0.0;
        int yIdx, uIdx, vIdx;
        bool exist = true;
        if( isInside( col0, row0, _width, _height) )
        {
            if (_step == 0) {
                yIdx = row0 * _width + col0;
                uIdx = ((row0 >> 1) * (_width >> 1) + (col0 >> 1)) * 2 + len;
                vIdx = uIdx + 1;
            } else {
                yIdx = row0 * _step + col0;
                uIdx = ((row0 >> 1) * (_step >> 1) + (col0 >> 1)) * 2 + _step * _height;
                vIdx = uIdx + 1;
            }

            yBi += w0 * (float)d_nv12[yIdx];
            uBi += w0 * (float)d_nv12[uIdx];
            vBi += w0 * (float)d_nv12[vIdx];
        }
        else
            exist = false;

        if( isInside( col1, row1, _width, _height) )
        {
            if (_step == 0) {
                yIdx = row1 * _width + col1;
                uIdx = ((row1 >> 1) * (_width >> 1) + (col1 >> 1)) * 2 + len;
                vIdx = uIdx + 1;
            } else {
                yIdx = row1 * _step + col1;
                uIdx = ((row1 >> 1) * (_step >> 1) + (col1 >> 1)) * 2 + _step * _height;
                vIdx = uIdx + 1;
            }

            yBi += w1 * (float)d_nv12[yIdx];
            uBi += w1 * (float)d_nv12[uIdx];
            vBi += w1 * (float)d_nv12[vIdx];
        }
        else
            exist = false;

        if( isInside( col2, row2, _width, _height) )
        {
            if (_step == 0) {
                yIdx = row2 * _width + col2;
                uIdx = ((row2 >> 1) * (_width >> 1) + (col2 >> 1)) * 2 + len;
                vIdx = uIdx + 1;
            } else {
                yIdx = row2 * _step + col2;
                uIdx = ((row2 >> 1) * (_step >> 1) + (col2 >> 1)) * 2 + _step * _height;
                vIdx = uIdx + 1;
            }

            yBi += w2 * (float)d_nv12[yIdx];
            uBi += w2 * (float)d_nv12[uIdx];
            vBi += w2 * (float)d_nv12[vIdx];
        }
        else
            exist = false;

        if( isInside( col3, row3, _width, _height) )
        {
            if (_step == 0) {
                yIdx = row3 * _width + col3;
                uIdx = ((row3 >> 1) * (_width >> 1) + (col3 >> 1)) * 2 + len;
                vIdx = uIdx + 1;
            } else {
                yIdx = row3 * _step + col3;
                uIdx = ((row3 >> 1) * (_step >> 1) + (col3 >> 1)) * 2 + _step * _height;
                vIdx = uIdx + 1;
            }

            yBi += w3 * (float)d_nv12[yIdx];
            uBi += w3 * (float)d_nv12[uIdx];
            vBi += w3 * (float)d_nv12[vIdx];
        }
        else
            exist = false;

        float y_ = (yBi + 0.5) - 16,
              u_ = (uBi + 0.5) - 128,
              v_ = (vBi + 0.5) - 128;

        // FIXME: 
        // float b = y_*_color_T[0] + v_*_color_T[1] + u_*_color_T[2];
        // float g = y_*_color_T[3] + v_*_color_T[4] + u_*_color_T[5];
        // float r = y_*_color_T[6] + v_*_color_T[7] + u_*_color_T[8];
        float b = y_               + 1.139  *u_;
        float g = y_ - 0.39465*v_  - 0.58060*u_;
        float r = y_ + 2.03211*v_              ;

        int bIdx, gIdx, rIdx;
        
        // vietth use BGR as input 
        bIdx = procIdx;
        gIdx = procIdx + cropLen;
        rIdx = procIdx + cropLen * 2;
        
        if( exist )
        {
            d_rgbBlob[rIdx] = (r < 0)? 0 : ((r > 255)? 255 : r);
            d_rgbBlob[gIdx] = (g < 0)? 0 : ((g > 255)? 255 : g);
            d_rgbBlob[bIdx] = (b < 0)? 0 : ((b > 255)? 255 : b);
        }
        else
        {
            d_rgbBlob[rIdx] = d_rgbBlob[gIdx] = d_rgbBlob[bIdx] = 0;
        }
        d_rgbBlob[rIdx] = (d_rgbBlob[rIdx] - 127.5) / 128.0;
        d_rgbBlob[gIdx] = (d_rgbBlob[gIdx] - 127.5) / 128.0;
        d_rgbBlob[bIdx] = (d_rgbBlob[bIdx] - 127.5) / 128.0;
    }
}

cudaError_t gpu_invWarpNV12ToRGBBlob( const unsigned char *d_nv12, float *d_rgbBlob, const float *d_T, const int _width, const int _height, const int d_step,
    const int _cropX, const int _cropY, const int _cropWidth, const int _cropHeight, cudaStream_t stream )
{
    if( d_nv12 == NULL || d_rgbBlob == NULL )
	{
		printf( "!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
		return cudaErrorInvalidDevicePointer;
	}

	if( (_width < 0) || (_height < 0) )
	{
		printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
		return cudaErrorInvalidValue;
	}
	if( (_cropWidth < 0) || (_cropHeight < 0) || (_cropX < 0) || (_cropY < 0) )
	{
		printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
		return cudaErrorInvalidValue;
	}

	int numel = _cropWidth * _cropHeight;
	cuda_invWarpNV12ToRGBBlob<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK, 0, stream>>>
			( d_nv12, d_rgbBlob, d_T, _width, _height, d_step, _cropX, _cropY, _cropWidth, _cropHeight, BT709);
    return cudaStreamSynchronize(stream);
}

/*=================================================
 *
 *  RGB Image Affine Warping
 *
 *===============================================*/
__global__ void cuda_invWarpRGBAToRGBBlob( const unsigned char *d_rgba, float *d_rgbBlob, const float *d_T, const int _width, const int _height, const int _step,
    const int _cropX, const int _cropY, const int _cropWidth, const int _cropHeight )
{
    // int len = _width * _height,
    int cropLen = _cropWidth * _cropHeight;
    int procIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if( procIdx < cropLen )
    {
        /**
         * NOTE: Compute index of pixel in d_rgba that is corresponding to current pixel in d_rgbBlob
         * 
         */
        int row  = procIdx / _cropWidth + _cropY,
            col  = procIdx % _cropWidth + _cropX;
        float x  = (float)col + 0.5,
              y  = (float)row + 0.5;
        float invX = d_T[0] * x + d_T[1] * y + d_T[2],
              invY = d_T[3] * x + d_T[4] * y + d_T[5],
              invW = d_T[6] * x + d_T[7] * y + d_T[8];
        if( invW )
        {
            invX /= invW;
            invY /= invW;
        }
        
        /**
         * NOTE: Compute index of 4 nearest neighbours of th imaginary source pixel
         * 
         */
        int row0 = (invY >= 0) ? (int)invY : ((int)invY - 1),
            col0 = (invX >= 0) ? (int)invX : ((int)invX - 1);

        float dx = invX - (float)col0 - 0.5,
              dy = invY - (float)row0 - 0.5;
        int dR   = (dy < 0) ? -1 : 1,
            dC   = (dx < 0) ? -1 : 1;

        int row1 = row0,
            col1 = col0 + dC,
            row2 = row0 + dR,
            col2 = col0 + dC,
            row3 = row0 + dR,
            col3 = col0;

        /**
         * NOTE: Compute bilinear weights
         * 
         */
        float alphaX = 1.0 - fabsf(dx),
              alphaY = 1.0 - fabsf(dy);
        float betaX  = 1.0 - alphaX,
              betaY  = 1.0 - alphaY;
        float w0 = alphaX * alphaY,
              w1 = betaX * alphaY,
              w2 = betaX * betaY,
              w3 = alphaX * betaY;

        /**
         * NOTE: Perform bilinear interpolation
         * 
         */
        float r = 0.0, g = 0.0, b = 0.0;
        int rgbIdx; // index in the origin 
        bool exist = true;
        if( isInside( col0, row0, _width, _height) )
        {
            if (_step == 0) {
                rgbIdx = row0 * _width + col0;
                rgbIdx *= 4;
            } else {
                rgbIdx = row0 * _step + col0 * 4;
            }

            r += w0 * (float)d_rgba[rgbIdx];
            g += w0 * (float)d_rgba[rgbIdx + 1];
            b += w0 * (float)d_rgba[rgbIdx + 2];
        }
        else
            exist = false;

        if( isInside( col1, row1, _width, _height) )
        {
            if (_step == 0) {
                rgbIdx = row1 * _width + col1;
                rgbIdx *= 4;
            } else {
                rgbIdx = row1 * _step + col1 * 4;
            }
            
            r += w1 * (float)d_rgba[rgbIdx];
            g += w1 * (float)d_rgba[rgbIdx + 1];
            b += w1 * (float)d_rgba[rgbIdx + 2];
        }
        else
            exist = false;

        if( isInside( col2, row2, _width, _height) )
        {
            if (_step == 0) {
                rgbIdx = row2 * _width + col2;
                rgbIdx *= 4;
            } else {
                rgbIdx = row2 * _step + col2 * 4;
            }
            
            r += w2 * (float)d_rgba[rgbIdx];
            g += w2 * (float)d_rgba[rgbIdx + 1];
            b += w2 * (float)d_rgba[rgbIdx + 2];
        }
        else
            exist = false;

        if( isInside( col3, row3, _width, _height) )
        {
            if (_step == 0) {
                rgbIdx = row3 * _width + col3;
                rgbIdx *= 4;
            } else {
                rgbIdx = row3 * _step + col3 * 4;
            }
            
            r += w3 * (float)d_rgba[rgbIdx];
            g += w3 * (float)d_rgba[rgbIdx + 1];
            b += w3 * (float)d_rgba[rgbIdx + 2];
        }
        else
            exist = false;

        int bIdx, gIdx, rIdx; // index in the blob
        // vietth use BGR as input 
        bIdx = procIdx;
        gIdx = procIdx + cropLen;
        rIdx = procIdx + cropLen * 2;


        if( exist )
        {
            d_rgbBlob[rIdx] = (r < 0)? 0 : ((r > 255)? 255 : r);
            d_rgbBlob[gIdx] = (g < 0)? 0 : ((g > 255)? 255 : g);
            d_rgbBlob[bIdx] = (b < 0)? 0 : ((b > 255)? 255 : b);
        }
        else
        {
            // Zero padding
            d_rgbBlob[rIdx] = d_rgbBlob[gIdx] = d_rgbBlob[bIdx] = 0;
        }

        /**
         * NOTE: normalization
         * 
         */
        d_rgbBlob[rIdx] = (d_rgbBlob[rIdx] - 127.5) / 128.0;
        d_rgbBlob[gIdx] = (d_rgbBlob[gIdx] - 127.5) / 128.0;
        d_rgbBlob[bIdx] = (d_rgbBlob[bIdx] - 127.5) / 128.0;
    }
}

cudaError_t gpu_invWarpRGBAToRGBBlob(const unsigned char *d_nv12, float *d_rgbBlob, const float *d_T, const int d_width, const int d_height, const int d_step,
    const int _cropX, const int _cropY, const int _cropWidth, const int _cropHeight,
    cudaStream_t stream)
{
    if( d_nv12 == NULL || d_rgbBlob == NULL )
	{
		printf("!Error: %s:%d: Invalid device pointers\n", __FUNCTION__, __LINE__ );
		return cudaErrorInvalidDevicePointer;
	}

    if( (d_width < 0) || (d_height < 0) )
	{
		printf( "!Error: %s:%d: Invalid image dimension\n", __FUNCTION__, __LINE__ );
		return cudaErrorInvalidValue;
	}

    if( (_cropWidth < 0) || (_cropHeight < 0) || (_cropX < 0) || (_cropY < 0) )
	{
		printf( "!Error: %s:%d: Invalid crop dimension\n", __FUNCTION__, __LINE__ );
		return cudaErrorInvalidValue;
	}

    int numel = _cropWidth * _cropHeight;
	cuda_invWarpRGBAToRGBBlob<<<(numel+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK, 0, stream>>>
			( d_nv12, d_rgbBlob, d_T, d_width, d_height, d_step, _cropX, _cropY, _cropWidth, _cropHeight );
    return cudaStreamSynchronize(stream);
}