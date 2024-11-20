#include "registration.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <iostream>

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

        // Point3D query_point = R * dev_pcs[index] + t;
        // const float query[] {query_point.x, query_point.y, query_point.z};

        // size_t nearest_index;
        // float distance_squared;
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

#if TEST_KDTREE
    __global__ void kernTestKDTreeLookUp(int N, Point3D query, FlattenedKDTree* fkdt, float* min_dists, size_t* min_indices)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        fkdt->find_nearest_neighbor(query, *min_dists, *min_indices);
    }
#endif
    
    __host__ float Registration::run(Rotation &q, glm::vec3 &t) 
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

#if TEST_KDTREE
        glm::vec3 queryPoint = this->pcs[1152];
        float bestDist = INF;
        size_t bestIndex;

        float* dev_best_dist;
        size_t* dev_best_idx;
        cudaMalloc((void**)&dev_best_dist, sizeof(float) * 10);
        cudaMalloc((void**)&dev_best_idx, sizeof(size_t) * 10);

        cudaMemcpy(dev_best_dist, &bestDist, sizeof(float), cudaMemcpyHostToDevice);

        kernTestKDTreeLookUp <<<1, 1>>> (1, queryPoint, dev_fkdt, dev_best_dist, dev_best_idx);

        cudaMemcpy(&bestDist, dev_best_dist, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bestIndex, dev_best_idx, sizeof(size_t), cudaMemcpyDeviceToHost);

        Logger(LogLevel::DEBUG) << "k-d tree on GPU:\t" << "best idx: " << bestIndex << "\tbest dist: " << bestDist;

        cudaFree(dev_best_dist);
        cudaFree(dev_best_idx);
#endif

        delete[] corr;

        return 0.0f;
    }


    //============================================
    //            Flattened k-d tree
    //============================================
    
    FlattenedKDTree::FlattenedKDTree(const KDTree& kdt, const PointCloud& pct)
    {
        thrust::host_vector<ArrayNode> h_array;
        thrust::host_vector<uint32_t> h_vAcc = kdt.vAcc_;  // Copy vAcc to host vector
        thrust::host_vector<Point3D> h_pct(pct.begin(), pct.end());

        // Convert KDTree to array on the host
        size_t currentIndex = 0;
        flatten_KDTree(kdt.root_node_, h_array, currentIndex);

        // Transfer to device
        d_array = h_array;
        d_vAcc = h_vAcc;
        d_pct = h_pct;
    }

    void FlattenedKDTree::flatten_KDTree(const KDTree::Node* root, thrust::host_vector<ArrayNode>& array, size_t& currentIndex)
    {
        if (root == nullptr) return;

        size_t index = currentIndex++;
        array.resize(index + 1);

        if (root->child1 == nullptr && root->child2 == nullptr) {
            // Leaf node
            array[index].is_leaf = true;
            array[index].data.leaf.left = root->node_type.lr.left;
            array[index].data.leaf.right = root->node_type.lr.right;
        }
        else {
            // Non-leaf node
            array[index].is_leaf = false;
            array[index].data.nonleaf.divfeat = root->node_type.sub.divfeat;
            array[index].data.nonleaf.divlow = root->node_type.sub.divlow;
            array[index].data.nonleaf.divhigh = root->node_type.sub.divhigh;

            // Recursively flatten left and right child nodes
            size_t child1Index = currentIndex;
            flatten_KDTree(root->child1, array, currentIndex);
            array[index].data.nonleaf.child1 = child1Index;

            size_t child2Index = currentIndex;
            flatten_KDTree(root->child2, array, currentIndex);
            array[index].data.nonleaf.child2 = child2Index;
        }
    }

    __device__ float distance_squared(const Point3D p1, const Point3D p2)
    {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ void FlattenedKDTree::find_nearest_neighbor(const Point3D query, size_t index, float &best_dist, size_t &best_idx, int depth)
    {
        if (index >= d_array.size()) return;

        const ArrayNode& node = d_array[index];
        if (node.is_leaf)
        {
            // Leaf node: Check all points in the leaf node
            size_t left = node.data.leaf.left;
            size_t right = node.data.leaf.right;
            for (size_t i = left; i <= right; i++)
            {
                float dist = distance_squared(query, d_pct[d_vAcc[i]]);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = d_vAcc[i];
                }
            }
        }
        else
        {
            // Non-leaf node: Determine which child to search
            int axis = node.data.nonleaf.divfeat;
            float diff = query[axis] - node.data.nonleaf.divlow;

            // Choose the near and far child based on comparison
            size_t nearChild = diff < 0 ? node.data.nonleaf.child1 : node.data.nonleaf.child2;
            size_t farChild = diff < 0 ? node.data.nonleaf.child2 : node.data.nonleaf.child1;

            // Search near child
            find_nearest_neighbor(query, nearChild, best_dist, best_idx, depth + 1);

            // Search far child if needed
            if (diff * diff < best_dist)
            {
                find_nearest_neighbor(query, farChild, best_dist, best_idx, depth + 1);
            }
        }
    }

}