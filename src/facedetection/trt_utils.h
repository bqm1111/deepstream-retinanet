#ifndef TRT_UTILS_H_pQBQkT4K4J
#define TRT_UTILS_H_pQBQkT4K4J

#include <iostream>
#include <fstream>
#include <map>
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "assert.h"

#define CHECK(ans)                                       \
	{                                                    \
		gpuAssert_ned3QSPqhX((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert_ned3QSPqhX(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUAssert: %s %s:%d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

namespace trt
{

	// Load weights from files shared with TensorRT samples
	// TensorRT weight files have a simple space delimited format:
	// [type] [size] <data x size in hex>
	// Loaded weights will be stored in host memory
	// Call freeWeights() after build the engine.
	inline std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file)
	{
		std::cout << "Loading weights: " << file << std::endl;
		std::map<std::string, nvinfer1::Weights> weightMap;

		// Open weights file
		std::ifstream input(file);
		assert(input.is_open() && "Unable to load weight file.");

		// Read number of weight blobs
		int32_t count;
		input >> count;
		assert(count > 0 && "Invalid weight map file.");

		while (count--)
		{
			nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
			uint32_t size;

			// Read name and type of blob
			std::string name;
			input >> name >> std::dec >> size;
			wt.type = nvinfer1::DataType::kFLOAT;

			// Load blob
			uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
			for (uint32_t x = 0, y = size; x < y; ++x)
			{
				input >> std::hex >> val[x];
			}
			wt.values = val;
			wt.count = size;
			weightMap[name] = wt;
		}

		return weightMap;
	}

	// free memory hold by Weights
	inline void freeWeights(std::map<std::string, nvinfer1::Weights> weightMap)
	{
		// Release host memory
		for (auto &mem : weightMap)
		{
			free((void *)(mem.second.values));
			mem.second.values = NULL;
		}
	}

#ifdef DEBUG
	int layer_count = 0;
	inline void printInfo(nvinfer1::ILayer *layer)
	{
		nvinfer1::Dims dim = layer->getOutput(0)->getDimensions();
		std::cout << std::right << std::setw(40) << layer->getName() << std::flush
				  << "-" << ++layer_count << " " << dim.nbDims << ": ";
		for (int i = 0; i < dim.nbDims; i++)
		{
			std::cout << dim.d[i] << ", ";
		}
		std::cout << std::endl;
	}
#else
	inline void printInfo(__attribute__((unused)) nvinfer1::ILayer *layer){};
#endif // DEBUG

	struct WeightNotFoundException : public std::runtime_error
	{
		WeightNotFoundException(const std::string &key)
			: std::runtime_error("Key " + key + " is not found in the weights") {}
	};

	static inline nvinfer1::Weights getWeights(const std::map<std::string, nvinfer1::Weights>& weightMap,
											   std::string key)
	{
		if (weightMap.count(key) != 1)
		{
			throw WeightNotFoundException(key);
			// exit(-1);
		}
		return weightMap.at(key);
	}

} // namespace trt

#endif // TRT_UTILS_H_pQBQkT4K4J
