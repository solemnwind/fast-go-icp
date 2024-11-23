#include "fgoicp.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <queue>

namespace icp
{
    void PointCloudRegistration::run()
    {
        branch_and_bound();
    }

}