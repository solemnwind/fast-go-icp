#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace icp
{
    using PointCloudDev = thrust::device_vector<Point3D>;
    __device__ float brute_force_find_nearest_neighbor(const Point3D query, const Point3D* d_pct, size_t nt);

    //============================================
    //                   LUT
    //============================================

    class NearestNeighborLUT
    {
    private:
        float definition;  // Definition of mesh size, default to 0.002: 500 * 500 * 500
        int3 dims;
        cudaArray* d_cudaArray;
        float* d_lutData;

    public:
        cudaTextureObject_t texObj;

        NearestNeighborLUT(size_t n = 500);  // TODO: needs fix
        ~NearestNeighborLUT();

        void build(const PointCloud& pct);

        __device__ float search(const float3 query) const;

    private:
        void initializeCudaTexture();
        void cleanupCudaTexture();
    };

    //============================================
    //                Registration
    //============================================

    class Registration
    {
    private:
        // Target point cloud
        const PointCloud& pct;
        // Source point cloud
        const PointCloud& pcs;
        //Number of target cloud points
        const size_t nt;
        // Number of source cloud points
        const size_t ns;
        // Target point cloud on device
        const PointCloudDev d_pct;
        // Source point cloud on device
        const PointCloudDev d_pcs;

        NearestNeighborLUT nnlut;
        NearestNeighborLUT* d_nnlut;

    public:
        Registration(const PointCloud &pct, size_t nt, const PointCloud &pcs, size_t ns) : 
            pct(pct), pcs(pcs),                     // init point clouds data (host)
            nt(nt), ns(ns),                         // init number of points
            d_pct(pct.begin(), pct.end()),          // init target point cloud (device)
            d_pcs(pcs.begin(), pcs.end()),          // init source point cloud (device)
            nnlut(300)
        {
            nnlut.build(pct);

            cudaMalloc((void**)&d_nnlut, sizeof(NearestNeighborLUT));
            cudaMemcpy(d_nnlut, &nnlut, sizeof(NearestNeighborLUT), cudaMemcpyHostToDevice);
        }

        ~Registration()
        {
            cudaFree(d_nnlut);
        }

        using BoundsResult_t = std::tuple<std::vector<float>, std::vector<float>>;

        /**
         * @brief Run Go-ICP registration algorithm.
         * 
         * @return float: MSE error
         */
        float compute_sse_error(glm::mat3 R, glm::vec3 t) const;
        BoundsResult_t compute_sse_error(RotNode &rnode, std::vector<TransNode>& tnodes, bool fix_rot, StreamPool& stream_pool) const;
    };

}

#endif // REGISTRATION_HPP