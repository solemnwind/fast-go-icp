#include "fgoicp.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <queue>

namespace icp
{
    __host__ void PointCloudRegistration::branch_and_bound()
    {
        // Outer BnB
        // Initialize rotation nodes
        std::priority_queue<RotNode> q_rnode;
        RotNode rnode_init {{0.0f, 0.0f, 0.0f}, 
                            1.0f, 
                            INF, 0.0f};
        
         
            // Invoke inner BnB
        
        // Prune        
    }
    
}