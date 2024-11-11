#include "kdtree.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace icp
{
    __device__ __host__ float KDTree::query(const Point3D* const point)
    {
        return 1.0f;
    }
}