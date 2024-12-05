#include "common.hpp"

namespace icp
{
    bool Logger::verbose_ = false;

    void cudaCheckError(string info, bool silent)
    {
#if CUDA_DEBUG
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            Logger(LogLevel::Error) << info << " failed: " << cudaGetErrorString(err);
        }
        else if (!silent)
        {
            Logger(LogLevel::Debug) << info << " success.";
        }
#endif
    }
}
