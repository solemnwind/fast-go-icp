#include "registration.hpp"
#include <thrust/device_vector.h>

namespace icp
{
    struct Correspondence
    {
        size_t idx_s;
        size_t idx_t;
        float dist;
    };
    
    __host__ __device__ float Registration::run(Rotation &q, Vector &t) 
    {
        size_t num_crpnds = min(nt, ns);

        

        for (size_t iter = 0; iter < max_iter; ++iter)
        {
            
        }
    }
}