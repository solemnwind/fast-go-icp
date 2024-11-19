#include "fgoicp/fgoicp.hpp"
#include <cuda.h>


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    // Set stack size to 16 KB to avoid stack overflow in recursion
    size_t new_stack_size = 16384;
    cudaError_t err = cudaThreadSetLimit(cudaLimitStackSize, new_stack_size);

    if (err != cudaSuccess) {
        printf("Error setting stack size: %s\n", cudaGetErrorString(err));
    }
    else {
        printf("Stack size successfully set to %zu bytes.\n", new_stack_size);
    }

    icp::PointCloudRegistration pcr(argv[1]);
    pcr.run();

    return 0;
}
