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

    __device__ float distanceSquared(const Point3D p1, const Point3D p2) {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ void find_nearest_neighbor(const Point3D target, size_t index, float* bestDist, size_t* bestIndex, int depth, ArrayNode* array, size_t size_array, uint32_t* vAcc_, Point3D* pct) {
        if (index >= size_array) return;

        const ArrayNode& node = array[index];
        if (node.is_leaf) 
        {
            // Leaf node: Check all points in the leaf node
            size_t left = node.data.leaf.left;
            size_t right = node.data.leaf.right;
            for (size_t i = left; i <= right; i++) 
            {
                float dist = distanceSquared(target, pct[vAcc_[i]]);
                if (dist < *bestDist) 
                {
                    *bestDist = dist;
                    *bestIndex = vAcc_[i];
                }
            }
        }
        else 
        {
            // Non-leaf node: Determine which child to search
            int axis = node.data.nonleaf.divfeat;
            float diff = target[axis] - node.data.nonleaf.divlow;

            // Choose the near and far child based on comparison
            size_t nearChild = diff < 0 ? node.data.nonleaf.child1 : node.data.nonleaf.child2;
            size_t farChild = diff < 0 ? node.data.nonleaf.child2 : node.data.nonleaf.child1;

            // Search near child
            find_nearest_neighbor(target, nearChild, bestDist, bestIndex, depth + 1, array, size_array, vAcc_, pct);

            // Search far child if needed
            if (diff * diff < *bestDist) 
            {
                find_nearest_neighbor(target, farChild, bestDist, bestIndex, depth + 1, array, size_array, vAcc_, pct);
            }
        }
    }

    __global__ void kernTestKDTreeLookUp(int N, Point3D query, float* min_dists, size_t* min_indices, ArrayNode* array, size_t size_array, uint32_t* vAcc_, Point3D* pct)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        find_nearest_neighbor(query, 0, min_dists, min_indices, 0, array, size_array, vAcc_, pct);
    }
    
    __host__ float Registration::run(Rotation &q, glm::vec3 &t) 
    {
        // size_t num_crpnds = ns;

        // Correspondence* corr = new Correspondence[num_crpnds];

        // float* mat_pct = new float[num_crpnds * 3];
        // float* mat_pcs = new float[num_crpnds * 3];

        // glm::vec3 target_centroid {0.0f, 0.0f, 0.0f};
        // glm::vec3 source_centroid {0.0f, 0.0f, 0.0f};

        // float error = 0;

        // for (size_t iter = 0; iter < max_iter; ++iter)
        // {
            
        // }

        // Test kd-tree
        glm::vec3 queryPoint = this->pcs[1152];
        std::cout << queryPoint.x << ", " << queryPoint.y << ", " << queryPoint.z << "\n";
        float query[3] = { queryPoint.x, queryPoint.y, queryPoint.z };

        size_t nearestIndex;
        float outDistSqr;
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&nearestIndex, &outDistSqr);

        this->kdt_target.findNeighbors(resultSet, query, nanoflann::SearchParameters(10));
        std::cout << nearestIndex << ", " << outDistSqr << "\n";

        // Test flattened kd-tree
        // auto flattened_kdt = convert_KDTree_to_array(this->kdt_target);
        // std::vector<ArrayNode> array;
        // size_t currentIndex = 0;
        // flatten_KDTree(kdt_target.root_node_, array, currentIndex);

        FlattenedKDTree fkdt {kdt_target, pct};

        ArrayNode* dev_array;
        uint32_t* dev_vAcc_;
        Point3D* dev_pct;

        cudaMalloc((void**)&dev_array, sizeof(ArrayNode) * fkdt.array.size());
        cudaMalloc((void**)&dev_vAcc_, sizeof(uint32_t) * fkdt.vAcc_.size());
        cudaMalloc((void**)&dev_pct, sizeof(Point3D) * pct.size());

        cudaMemcpy(dev_array, fkdt.array.data(), sizeof(ArrayNode) * fkdt.array.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vAcc_, fkdt.vAcc_.data(), sizeof(uint32_t) * fkdt.vAcc_.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_pct, pct.data(), sizeof(Point3D) * pct.size(), cudaMemcpyHostToDevice);

        float bestDist = INF;
        size_t bestIndex;
        // fkdt.find_nearest_neighbor(queryPoint, bestDist, bestIndex);

        float* dev_best_dist;
        size_t* dev_best_idx;
        cudaMalloc((void**)&dev_best_dist, sizeof(float) * 10);
        cudaMalloc((void**)&dev_best_idx, sizeof(size_t) * 10);

        cudaMemcpy(dev_best_dist, &bestDist, sizeof(float), cudaMemcpyHostToDevice);

        kernTestKDTreeLookUp <<<1, 1>>> (1, queryPoint, dev_best_dist, dev_best_idx, dev_array, fkdt.array.size(), dev_vAcc_, dev_pct);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "CUDA error");
            fprintf(stderr, ": %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(&bestDist, dev_best_dist, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bestIndex, dev_best_idx, sizeof(size_t), cudaMemcpyDeviceToHost);

        std::cout << "best idx: " << bestIndex << " best dist: " << bestDist << "\n";

        cudaFree(dev_best_dist);
        cudaFree(dev_best_idx);

        cudaFree(dev_array);
        cudaFree(dev_pct);
        cudaFree(dev_vAcc_);

        // delete[] corr;
        return 0.0f;
    }
}