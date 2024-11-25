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


    public:
        Registration(const PointCloud &pct, size_t nt, const PointCloud &pcs, size_t ns) : 
            pct(pct), pcs(pcs),                     // init point clouds data (host)
            nt(nt), ns(ns),                         // init number of points
            d_pct(pct.begin(), pct.end()),          // init target point cloud (device)
            d_pcs(pcs.begin(), pcs.end())           // init source point cloud (device)
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