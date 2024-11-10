#include "fgoicp.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>

namespace icp
{
    /**
     * @brief Perform branch-and-bound algorithm in SE(3) space
     * 
     * @details This is a __host__ function, 
     * it launches cuda kernels for translation BnB.
     * 
     * @return none
     */
    __host__ void PointCloudRegistration::branch_and_bound()
    {
        // Outer BnB
        // Initialize rotation nodes
            // Invoke inner BnB
        
        // Prune        
    }
    
}