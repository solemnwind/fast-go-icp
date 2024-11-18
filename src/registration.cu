#include "registration.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

namespace icp
{
    struct Correspondence
    {
        size_t idx_s;
        size_t idx_t;
        float dist_squared;
        Point3D ps_transformed;
    };

    __global__ void kernFindNearestNeighbor(int N, glm::mat3 R, glm::vec3 t, const Point3D* dev_pcs, const KDTree* dev_kdt, Correspondence* dev_corrs)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        Point3D query_point = R * dev_pcs[index] + t;
        const float query[] {query_point.x, query_point.y, query_point.z};

		size_t nearest_index;
		float distance_squared;
		// nanoflann::KNNResultSet<float> result(1);
		// result.init(&nearest_index, &distance_squared);
        // dev_kdt->findNeighbors(result, query, nanoflann::SearchParameters(1));

        // dev_corrs[index].dist_squared = result;
        // dev_corrs[index].idx_s = index;
        // dev_corrs[index].idx_t = ret.idx_target;
        // dev_corrs[index].ps_transformed = query_point;
    }

    __global__ void kernSetRegistrationMatrices(int N, Rotation q, glm::vec3 t, const Point3D* dev_pcs, const Point3D* dev_pct, const Correspondence* dev_corrs, float* dev_mat_pcs, float* dev_mat_pct)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        Correspondence corr = dev_corrs[index];
        Point3D pt = dev_pct[corr.idx_t];
        Point3D ps = corr.ps_transformed;

        size_t mat_idx = index * 3;

        dev_mat_pct[mat_idx] = pt.x;
        dev_mat_pct[mat_idx + 1] = pt.y;
        dev_mat_pct[mat_idx + 2] = pt.z;

        dev_mat_pcs[mat_idx] = ps.x;
        dev_mat_pcs[mat_idx + 1] = ps.y;
        dev_mat_pcs[mat_idx + 2] = ps.z;
    }
    
    __host__ __device__ float Registration::run(Rotation &q, glm::vec3 &t) 
    {
        size_t num_crpnds = ns;

        Correspondence* corr = new Correspondence[num_crpnds];

        float* mat_pct = new float[num_crpnds * 3];
        float* mat_pcs = new float[num_crpnds * 3];

        glm::vec3 target_centroid {0.0f, 0.0f, 0.0f};
        glm::vec3 source_centroid {0.0f, 0.0f, 0.0f};

        float error = 0;

        for (size_t iter = 0; iter < max_iter; ++iter)
        {
            
        }

        delete[] corr;
        return 0.0f;
    }
}