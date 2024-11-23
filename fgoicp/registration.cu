#include "registration.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <queue>

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
        float* dev_errors;
        cudaMalloc((void**)&dev_errors, sizeof(float) * ns);

        size_t block_size = 256;
        dim3 threads_per_block(block_size);
        dim3 blocks_per_grid((ns + block_size - 1) / block_size);
        kernComputeRegistrationError <<<blocks_per_grid, threads_per_block>>> (
            ns, 
            glm::mat3(1.0f),    // Identity Rotation
            glm::vec3(0.f),     // Identity Translation
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

        branch_and_bound_SO3();

        return 0.0f;
    }

    float Registration::branch_and_bound_SO3()
    {
        // Initialize Rotation Nodes
        std::vector<std::unique_ptr<RotNode>> rcandidates;
        {
            constexpr float N = 8.0f;
            constexpr float step = 2.0f / N;
            constexpr float start = -1.0f + step;
            float span = 1.0f / 8.0f;
            for (float x = start; x < 1.0f; x += step)
            {
                for (float y = start; y < 1.0f; y += step)
                {
                    for (float z = start; z < 1.0f; z += step)
                    {
                        std::unique_ptr<RotNode> p_rnode = std::make_unique<RotNode>(x, y, z, span, M_INF, 0.0f);
                        if (p_rnode->overlaps_SO3()) { rcandidates.push_back(std::move(p_rnode)); }
                    }
                }
            }
        }

        while (true)
        {
            for (auto& p_rnode : rcandidates)
            {
                // BnB in R3 
                ResultBnBR3 res = branch_and_bound_R3(p_rnode->q);
                if (res.error < best_error)
                {
                    best_error = res.error;
                    best_rotation = p_rnode->q;
                    best_translation = res.translation;
                }
                if (res.error < sse_threshold)
                {
                    return res.error;
                }
            }
            break;
        }
        return 0.0f;
    }

    Registration::ResultBnBR3 Registration::branch_and_bound_R3(Rotation q)
    {
        // TODO:  Need target point cloud stats
        // Assume the the target point cloud is within AABB [-2.0, -2.0, -1.0, 2.0, 2.0, 1.0]
        float xmin = -2.0;
        float ymin = -2.0;
        float zmin = -1.0;
        float xmax = 2.0;
        float ymax = 2.0;
        float zmax = 1.0;

        // Initialize
        std::queue<RotNode> tcandidates;    // Consider using priority queue
        {
            float step = 1.0f / 4.0f;
            float span = 1.0f / 8.0f;
            for (float x = xmin + span; x < xmax; x += step)
            {
                for (float y = ymin + span; y < ymax; y += step)
                {
                    for (float z = zmin + span; z < zmax; z += step)
                    {
                        tcandidates.emplace(x, y, z, span, M_INF, 0.0f);
                    }
                }
            }
        }

        while (true)
        {
            break;
        }

        return { 0.0f, {0.0f, 0.0f, 0.0f} };
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