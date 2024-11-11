#define GLM_FORCE_CUDA
#include "kernel.h"

extern int numPoints;
extern int numDataPoints;
extern int numModelPoints;

extern glm::vec3* dev_pos;
extern glm::vec3* dev_col;

extern glm::vec3* dev_dataBuffer;
extern glm::vec3* dev_modelBuffer;
extern glm::vec3* dev_corrBuffer;

extern glm::vec3* dev_centeredCorrBuffer;
extern glm::vec3* dev_centeredDataBuffer;
extern glm::mat3* dev_ABtBuffer;

// Helper Functions
void checkCUDAError(const char* msg, int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// Helper kernel functions
__global__ void kernResetVec3Buffer(int N, glm::vec3* intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

__global__ void kernCopyPositionsToVBO(int N, glm::vec3* pos, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyColorsToVBO(int N, glm::vec3* col, float* vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = col[index].x + 0.3f;
		vbo[4 * index + 1] = col[index].y + 0.3f;
		vbo[4 * index + 2] = col[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

void PointCloud::initBuffers(std::vector<glm::vec3>& dataBuffer, std::vector<glm::vec3>& modelBuffer) {
	// Use unified memory
	cudaMallocManaged((void**)&dev_pos, numPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_pos failed!");
	cudaMallocManaged((void**)&dev_col, numPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_col failed!");
	cudaMallocManaged((void**)&dev_dataBuffer, numDataPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_dataBuffer failed!");
	cudaMallocManaged((void**)&dev_modelBuffer, numModelPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_modelBuffer failed!");
	cudaMallocManaged((void**)&dev_corrBuffer, numDataPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_corrBuffer failed!");
	cudaMallocManaged((void**)&dev_centeredDataBuffer, numDataPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_centeredDataBuffer failed!");
	cudaMallocManaged((void**)&dev_centeredCorrBuffer, numDataPoints * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_centeredCorrBuffer failed!");
	cudaMallocManaged((void**)&dev_ABtBuffer, numDataPoints * sizeof(glm::mat3));
	checkCUDAErrorWithLine("cudaMallocManaged dev_ABtBuffer failed!");

	// Set Posistion Buffer
	std::copy(dataBuffer.begin(), dataBuffer.end(), dev_dataBuffer);
	std::copy(modelBuffer.begin(), modelBuffer.end(), dev_modelBuffer);
	std::copy(modelBuffer.begin(), modelBuffer.end(), dev_pos);
	std::copy(dataBuffer.begin(), dataBuffer.end(), dev_pos + numModelPoints);

	// Set color buffer
	dim3 dataBlocksPerGrid((numDataPoints + blockSize - 1) / blockSize);
	dim3 modelBlocksPerGrid((numModelPoints + blockSize - 1) / blockSize);
	kernResetVec3Buffer << <modelBlocksPerGrid, blockSize >> > (numModelPoints, dev_col, glm::vec3(0, 0, 1));
	kernResetVec3Buffer << < dataBlocksPerGrid, blockSize >> > (numDataPoints, &dev_col[numModelPoints], glm::vec3(1, 0, 0));

	cudaDeviceSynchronize();
}

void PointCloud::copyPointsToVBO(float* vbodptr_positions, float* vbodptr_colors) {
	dim3 fullBlocksPerGrid((numPoints + blockSize - 1) / blockSize);
	kernCopyPositionsToVBO <<<fullBlocksPerGrid, blockSize >> > (numPoints, dev_pos, vbodptr_positions, scene_scale);
	kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> > (numPoints, dev_col, vbodptr_colors, scene_scale);
	checkCUDAErrorWithLine("copyPointsToVBO failed!");

	cudaDeviceSynchronize();
}

void PointCloud::cleanupBuffers() {
	cudaFree(dev_pos);
	cudaFree(dev_col);
	cudaFree(dev_modelBuffer);
	cudaFree(dev_dataBuffer);
	cudaFree(dev_corrBuffer);
	cudaFree(dev_centeredDataBuffer);
	cudaFree(dev_centeredCorrBuffer);
	cudaFree(dev_ABtBuffer);

	checkCUDAErrorWithLine("cudaFree failed!");
}