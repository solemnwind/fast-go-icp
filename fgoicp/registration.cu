#include "registration.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

#define CUDA_DEBUG 1

namespace icp
{
    void cudaCheckError(string info, bool silent = true)
    {
#if CUDA_DEBUG
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            Logger(LogLevel::Error) << info << " failed: " << cudaGetErrorString(err);
        }
        else if (!silent)
        {
            Logger(LogLevel::Debug) << info << " success.";
        }
#endif
    }

    struct Correspondence
    {
        size_t idx_s;
        size_t idx_t;
        float dist_squared;
        Point3D ps_transformed;
    };

    __global__ void kernFindNearestNeighbor(int N, glm::mat3 R, glm::vec3 t, const Point3D* dev_pcs, const FlattenedKDTree* d_fkdt, Correspondence* dev_corrs)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        Point3D query_point = R * dev_pcs[index] + t;

        size_t nearest_index = 0;

        float distance_squared = M_INF;
        d_fkdt->find_nearest_neighbor(query_point, distance_squared, nearest_index);

        dev_corrs[index].dist_squared = distance_squared;
        dev_corrs[index].idx_s = index;
        dev_corrs[index].idx_t = nearest_index;
        dev_corrs[index].ps_transformed = query_point;
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

    __global__ void kernComputeRegistrationError(int N, glm::mat3 R, glm::vec3 t, const Point3D *d_pcs, const FlattenedKDTree* d_fkdt, float* d_errors)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        Point3D source_point = d_pcs[index];
        Point3D query_point = R * source_point + t;

        size_t nearest_index = 0;
        float distance_squared = M_INF;
        d_fkdt->find_nearest_neighbor(query_point, distance_squared, nearest_index);

        d_errors[index] = distance_squared;
    }
    
    float Registration::run(Rotation &q, glm::vec3 &t) 
    {
        // Initialize with no rotation
        RotNode init_rnode{ {0.0f, 0.0f, 0.0f}, 1.0f, 0.0f, 0.0f };

        float* dev_errors;
        cudaMalloc((void**)&dev_errors, sizeof(float) * ns);

        size_t block_size = 256;
        dim3 threads_per_block(block_size);
        dim3 blocks_per_grid((ns + block_size - 1) / block_size);
        kernComputeRegistrationError <<<blocks_per_grid, threads_per_block>>> (
            ns, 
            init_rnode.q.R, 
            glm::vec3(0.f, 0.f, 0.f), 
            thrust::raw_pointer_cast(d_pcs.data()),
            d_fkdt, 
            dev_errors);
        cudaDeviceSynchronize();
        cudaCheckError("Kernel launch", false);

        thrust::device_ptr<float> dev_errors_ptr(dev_errors);
        best_error = thrust::reduce(dev_errors_ptr, dev_errors_ptr + ns, 0.0f);
        cudaCheckError("thrust::reduce", false);
        Logger(LogLevel::Info) << "Initial Error: " << best_error;
        
        cudaFree(dev_errors);


        best_rnode = init_rnode;

        return 0.0f;
    }

    float branch_and_bound_SO3(RotNode& rnode)
    {
        return 0.0f;
    }


    //============================================
    //            Flattened k-d tree
    //============================================
    
    FlattenedKDTree::FlattenedKDTree(const KDTree& kdt, const PointCloud& pct) :
        h_vAcc{kdt.vAcc_},
        h_pct{pct.begin(), pct.end()}
    {
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

    __device__ __host__ float distance_squared(const Point3D p1, const Point3D p2)
    {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ __host__ void FlattenedKDTree::find_nearest_neighbor(const Point3D query, size_t index, float &best_dist, size_t &best_idx, int depth) const
    {
#ifdef  __CUDA_ARCH__
        if (index >= d_array.size()) return;
        const ArrayNode& node = d_array[index];
#else
        if (index >= h_array.size()) return;
        const ArrayNode& node = h_array[index]; 
#endif
        if (node.is_leaf)
        {
            // Leaf node: Check all points in the leaf node
            size_t left = node.data.leaf.left;
            size_t right = node.data.leaf.right;
            for (size_t i = left; i <= right; i++)
            {
#ifdef __CUDA_ARCH__
                float dist = distance_squared(query, d_pct[d_vAcc[i]]);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = d_vAcc[i];
                }
#else
                float dist = distance_squared(query, h_pct[h_vAcc[i]]);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = h_vAcc[i];
                }
#endif
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