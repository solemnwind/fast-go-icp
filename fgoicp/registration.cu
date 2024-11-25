#include "registration.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>


namespace icp
{
    __global__ void kernComputeClosestError(int N, glm::mat3 R, glm::vec3 t, const Point3D *d_pcs, const FlattenedKDTree* d_fkdt, float* d_errors)
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

    __global__ void kernComputeBounds(int N, RotNode rnode, TransNode tnode, bool fix_rot, const Point3D* d_pcs, const FlattenedKDTree* d_fkdt, float* d_rot_ub_trans_ub, float* d_rot_ub_trans_lb)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        Point3D source_point = d_pcs[index];
        float trans_uncertain_radius = M_SQRT3 * tnode.span;
        Point3D query_point = rnode.q.R * source_point + tnode.t;

        float rot_uncertain_radius;
        if (!fix_rot)
        {
            float radius = source_point.x * source_point.x +
                source_point.y * source_point.y +
                source_point.z * source_point.z;
            float half_angle = rnode.span * M_SQRT3 * M_PI / 2.0f;  // TODO: Need examination, since we are using quaternions
            rot_uncertain_radius = 2.0f * radius * sin(half_angle);
        }

        size_t nearest_index = 0;
        float distance_squared = M_INF;
        d_fkdt->find_nearest_neighbor(query_point, distance_squared, nearest_index);

        float distance = sqrt(distance_squared);
        if (!fix_rot)
        {
            distance -= rot_uncertain_radius;
        }

        d_rot_ub_trans_ub[index] = distance > 0.0f ? distance * distance : 0.0f;


        float rot_ub_trans_lb = distance - trans_uncertain_radius;
        rot_ub_trans_lb = rot_ub_trans_lb > 0.0f ? rot_ub_trans_lb * rot_ub_trans_lb : 0.0f;
        d_rot_ub_trans_lb[index] = rot_ub_trans_lb;
    }
    
    float Registration::compute_sse_error(glm::mat3 R, glm::vec3 t) const
    {
        float* dev_errors;
        cudaMalloc((void**)&dev_errors, sizeof(float) * ns);

        size_t block_size = 32;
        dim3 threads_per_block(block_size);
        dim3 blocks_per_grid((ns + block_size - 1) / block_size);
        kernComputeClosestError <<<blocks_per_grid, threads_per_block>>> (
            ns, R, t,
            thrust::raw_pointer_cast(d_pcs.data()),
            d_fkdt,
            dev_errors);
        cudaDeviceSynchronize();
        cudaCheckError("Kernel launch");

        // Sum up the squared errors with thrust::reduce
        thrust::device_ptr<float> dev_errors_ptr(dev_errors);
        float sse_error = thrust::reduce(dev_errors_ptr, dev_errors_ptr + ns, 0.0f, thrust::plus<float>());
        cudaCheckError("thrust::reduce");

        cudaFree(dev_errors);

        return sse_error;
    }

    Registration::BoundsResult_t Registration::compute_sse_error(RotNode &rnode, std::vector<TransNode> &tnodes, bool fix_rot, StreamPool& stream_pool) const
    {
        size_t num_transforms = tnodes.size();
        std::vector<float> sse_rot_ub_trans_ub(num_transforms);
        std::vector<float> sse_rot_ub_trans_lb(num_transforms);

        // Allocate memory on the device for the errors for each (R, t) pair
        float* d_rot_ub_trans_ub;
        float* d_rot_ub_trans_lb;
        cudaMalloc((void**)&d_rot_ub_trans_ub, sizeof(float) * ns * num_transforms);
        cudaMalloc((void**)&d_rot_ub_trans_lb, sizeof(float) * ns * num_transforms);

        thrust::device_ptr<float> d_thrust_rot_ub_trans_ub(d_rot_ub_trans_ub);
        thrust::device_ptr<float> d_thrust_rot_ub_trans_lb(d_rot_ub_trans_lb);

        // Kernel launching parameters
        size_t block_size = 32;
        dim3 threads_per_block(block_size);
        dim3 blocks_per_grid((ns + block_size - 1) / block_size);

        // Launch kernel for each (R, t) pair on separate streams
        for (size_t i = 0; i < num_transforms; ++i) {
            // Get the appropriate stream from the stream pool
            cudaStream_t stream = stream_pool.getStream(i);

            // Launch the kernel with each (R, t) on a different stream
            kernComputeBounds <<<blocks_per_grid, threads_per_block, 0, stream>>> (
                ns, rnode, tnodes[i], fix_rot,
                thrust::raw_pointer_cast(d_pcs.data()),
                d_fkdt,
                d_rot_ub_trans_ub + i * ns,
                d_rot_ub_trans_lb + i * ns);
        }

        // Reduce the lower/upper bounds for each pair
        for (size_t i = 0; i < num_transforms; ++i) {
            // Thrust reduce launching parameters
            auto thrust_policy = thrust::cuda::par.on(stream_pool.getStream(i));

            sse_rot_ub_trans_ub[i] = thrust::reduce(
                thrust_policy,
                d_thrust_rot_ub_trans_ub + i * ns,
                d_thrust_rot_ub_trans_ub + (i + 1) * ns,
                0.0f,
                thrust::plus<float>()
            );

            sse_rot_ub_trans_lb[i] = thrust::reduce(
                thrust_policy,
                d_thrust_rot_ub_trans_lb + i * ns,
                d_thrust_rot_ub_trans_lb + (i + 1) * ns,
                0.0f,
                thrust::plus<float>()
            );
        }

        cudaDeviceSynchronize();

        // Free the device memory
        cudaFree(d_rot_ub_trans_ub);
        cudaFree(d_rot_ub_trans_lb);

        return { sse_rot_ub_trans_lb, sse_rot_ub_trans_ub };
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