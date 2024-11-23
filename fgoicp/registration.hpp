#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"
#include "kdtree_adaptor.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace icp
{
    using PointCloudDev = thrust::device_vector<Point3D>;

    //============================================
    //            Flattened k-d tree
    //============================================

    class FlattenedKDTree
    {
    private:
        struct ArrayNode
        {
            bool is_leaf;
            
            // Leaf or Non-leaf node data
            union {
                // Leaf node data
                struct {
                    size_t left, right;  // Indices of points in the leaf node
                } leaf;

                // Non-leaf node data
                struct {
                    int32_t divfeat;       // Dimension used for subdivision
                    float divlow, divhigh; // Range values used for subdivision
                    size_t child1, child2; // Indices of child nodes in the array
                } nonleaf;
            } data;
        };

        thrust::host_vector<ArrayNode> h_array;
        thrust::host_vector<uint32_t> h_vAcc;
        thrust::host_vector<Point3D> h_pct;

        // TODO: move them to Texture for better performance
        thrust::device_vector<ArrayNode> d_array;  // Flattened KD-tree on device
        thrust::device_vector<uint32_t> d_vAcc;    // Indices mapping
        thrust::device_vector<Point3D> d_pct;      // Point cloud on device

    public:
        FlattenedKDTree(const KDTree& kdt, const PointCloud& pct);

        /**
         * @brief  Finds the nearest neighbor with the flattened k-d tree.
         * @param  query      point in the source point cloud
         * @param  best_dist  shortest distance found
         * @param  best_idx   index of the nearest point in the target point cloud
         */
        __device__ __host__ void find_nearest_neighbor(const Point3D query, float& best_dist, size_t& best_idx) const
        {
            find_nearest_neighbor(query, 0, best_dist, best_idx, 0);
        }

    private:
        void flatten_KDTree(const KDTree::Node* root, thrust::host_vector<ArrayNode>& array, size_t& currentIndex);

        __device__ __host__ void find_nearest_neighbor(const Point3D query, size_t index, float& best_dist, size_t& best_idx, int depth) const;
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

        FlattenedKDTree* h_fkdt;
        FlattenedKDTree* d_fkdt;

        size_t max_iter;
        float best_error;
        Rotation best_rotation;
        glm::vec3 best_translation;

        // MSE threshold depends on the source point cloud stats.
        // If we normalize the source point cloud into a standard cube,
        // The MSE threshold can be specified without considering 
        // the point cloud stats.
        float mse_threshold;
        // SSE threshold is the summed error threshold,
        // the registration is considered converged if SSE threshold is reached.
        // If no trimming, sse_threshold = ns * mse_threshold
        float sse_threshold;

    public:
        Registration(const PointCloud &pct, size_t nt, const PointCloud &pcs, size_t ns) : 
            pct(pct), pcs(pcs),                     // init point clouds data (host)
            nt(nt), ns(ns),                         // init number of points
            d_pct(pct.begin(), pct.end()),          // init target point cloud (device)
            d_pcs(pcs.begin(), pcs.end()),          // init source point cloud (device)
            max_iter(10), best_error(M_INF),        // 
            mse_threshold(1E-3f),                   // init *mean* squared error threshold 
            sse_threshold(ns * mse_threshold)       // init *sum* of squared error threshold, determines convergence
        {
            // Create and build the KDTree
            PointCloudAdaptor pct_adaptor(pct);
            KDTree kdt_target(3, pct_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            kdt_target.buildIndex();

            // Flatten k-d tree and copy to device
            h_fkdt = new FlattenedKDTree(kdt_target, pct);
            cudaMalloc((void**)&d_fkdt, sizeof(FlattenedKDTree));
            cudaMemcpy(d_fkdt, h_fkdt, sizeof(FlattenedKDTree), cudaMemcpyHostToDevice);

            Logger(LogLevel::Info) << "KD-tree built with " << pct.size() << " points";
        }

        ~Registration()
        {
            delete h_fkdt;
            cudaFree(d_fkdt);
        }

        /**
         * @brief Run Go-ICP registration algorithm.
         * 
         * @return float: MSE error
         */
        float run(Rotation &q, glm::vec3 &t);

    private:
        struct ResultBnBR3
        {
            float error;
            glm::vec3 translation;
        };

        /**
         * @brief Perform branch-and-bound algorithm in Rotation Space SO(3)
         * 
         * @return float
         */
        float branch_and_bound_SO3();

        /**
         * @brief Perform branch-and-bound algorithm in Translation Space R(3)
         * 
         * @param 
         * @return ResultBnBR3 
         */
        ResultBnBR3 branch_and_bound_R3(Rotation q);
    };

}

#endif // REGISTRATION_HPP