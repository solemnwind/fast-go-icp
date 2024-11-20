#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"
#include "kdtree_adaptor.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define TEST_KDTREE 1

namespace icp
{
    //============================================
    //            Flattened k-d tree
    //============================================

    class FlattenedKDTree
    {
    public:
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

        thrust::device_vector<ArrayNode> d_array;  // Flattened KD-tree on device
        thrust::device_vector<uint32_t> d_vAcc;    // Indices mapping
        thrust::device_vector<Point3D> d_pct;      // Point cloud on device

        FlattenedKDTree(const KDTree& kdt, const PointCloud& pct);

        /**
         * @brief  Finds the nearest neighbor with the flattened k-d tree.
         * @param  query      point in the source point cloud
         * @param  best_dist  shortest distance found
         * @param  best_idx   index of the nearest point in the target point cloud
         */
        __device__ void find_nearest_neighbor(const Point3D query, float& best_dist, size_t& best_idx)
        {
            find_nearest_neighbor(query, 0, best_dist, best_idx, 0);
        }

    private:
        void flatten_KDTree(const KDTree::Node* root, thrust::host_vector<ArrayNode>& array, size_t& currentIndex);

        __device__ void find_nearest_neighbor(const Point3D query, size_t index, float& best_dist, size_t& best_idx, int depth);
    };

    //============================================
    //                Registration
    //============================================

    class Registration
    {
    public:
        Registration(const PointCloud &pct, size_t nt, const PointCloud &pcs, size_t ns) : 
            pct(pct), pcs(pcs),
            nt(nt), ns(ns)
        {
            // Create and build the KDTree
            PointCloudAdaptor pct_adaptor(pct);
            KDTree kdt_target(3, pct_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            kdt_target.buildIndex();

            // Flatten k-d tree and copy to device
            FlattenedKDTree fkdt(kdt_target, pct);
            cudaMalloc((void**)&dev_fkdt, sizeof(FlattenedKDTree));
            cudaMemcpy(dev_fkdt, &fkdt, sizeof(FlattenedKDTree), cudaMemcpyHostToDevice);

            Logger(LogLevel::INFO) << "KD-tree built with " << pct.size() << " points";

#if TEST_KDTREE
            // Test kd-tree
            glm::vec3 queryPoint = this->pcs[1152];
            float query[3] = { queryPoint.x, queryPoint.y, queryPoint.z };

            size_t nearestIndex;
            float outDistSqr;
            nanoflann::KNNResultSet<float> resultSet(1);
            resultSet.init(&nearestIndex, &outDistSqr);

            kdt_target.findNeighbors(resultSet, query, nanoflann::SearchParameters(10));
            Logger(LogLevel::DEBUG) << "k-d tree on CPU:\t" << "best idx: " << nearestIndex << " best dist: " << outDistSqr;
#endif
        }

        ~Registration()
        {
            // free kdt_target;
        }

        /**
         * @brief Run single-step ICP registration algorithm.
         * 
         * @return float: MSE error
         */
        float run(Rotation &q, glm::vec3 &t);

    private:
        // Target point cloud
        const PointCloud &pct;
        // Source point cloud
        const PointCloud &pcs;
        //Number of target cloud points
        const size_t nt;
        // Number of source cloud points
        const size_t ns;

        FlattenedKDTree* dev_fkdt;

        size_t max_iter = 10;

    };

}

#endif // REGISTRATION_HPP