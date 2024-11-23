#include "registration.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>


namespace icp
{
    //========================================================================================
    //                                     Registration
    //========================================================================================
    __global__ void kernComputeClosetError(int N, glm::mat3 R, glm::vec3 t, const Point3D *d_pcs, const FlattenedKDTree* d_fkdt, float* d_errors)
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
    
    float Registration::compute_sse_error(glm::mat3 R, glm::vec3 t) const
    {
        float* dev_errors;
        cudaMalloc((void**)&dev_errors, sizeof(float) * ns);

        size_t block_size = 32;
        dim3 threads_per_block(block_size);
        dim3 blocks_per_grid((ns + block_size - 1) / block_size);
        kernComputeClosetError <<<blocks_per_grid, threads_per_block>>> (
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

    std::vector<float> Registration::compute_sse_error(std::vector<glm::mat3> Rs, std::vector<glm::vec3> ts, StreamPool& stream_pool) const
    {
        // Ensure the input vectors are the same size
        assert(Rs.size() == ts.size());
        size_t num_matrices = Rs.size();

        // Allocate memory on the device for the errors for each (R, t) pair
        float* dev_errors;
        cudaMalloc((void**)&dev_errors, sizeof(float) * ns * num_matrices);

        size_t block_size = 32;
        dim3 threads_per_block(block_size);
        dim3 blocks_per_grid((ns + block_size - 1) / block_size);

        // Launch kernel for each (R, t) pair on separate streams
        for (size_t i = 0; i < num_matrices; ++i) 
        {
            // Get the appropriate stream from the stream pool
            cudaStream_t stream = stream_pool.getStream(i);

            // Launch the kernel with each (R, t) on a different stream
            kernComputeClosetError <<<blocks_per_grid, threads_per_block, 0, stream>>> (
                ns, Rs[i], ts[i],
                thrust::raw_pointer_cast(d_pcs.data()),
                d_fkdt,
                dev_errors + i * ns);  // Offset errors for each (R, t)
        }

        // Ensure kernel execution is correctly handled (wait for all streams)
        cudaDeviceSynchronize();  // Ensure all kernels finish before continuing
        cudaCheckError("Kernel launch");

        // Use thrust to compute the SSE error for each (R, t) pair
        auto thrust_policy = thrust::cuda::par.on(stream_pool.getStream(0));  // Using the first stream for reduction
        thrust::device_ptr<float> dev_errors_ptr(dev_errors);
        std::vector<float> sse_errors(num_matrices);

        // Reduce the errors for each pair
        for (size_t i = 0; i < num_matrices; ++i) 
        {
            sse_errors[i] = thrust::reduce(
                thrust_policy,
                dev_errors_ptr + i * ns,
                dev_errors_ptr + (i + 1) * ns,
                0.0f,
                thrust::plus<float>()
            );
        }

        cudaCheckError("thrust::reduce");

        // Free the device memory
        cudaFree(dev_errors);

        return sse_errors;
    }


    //========================================================================================
    //                                  Flattened k-d tree
    //========================================================================================
    
    FlattenedKDTree::FlattenedKDTree(const KDTree& kdt, const PointCloud& pct) :
        h_vAcc{ kdt.vAcc_ },
        h_pct{pct.begin(), pct.end()}
    {
        //h_pct.reserve(pct.size());
        //for (size_t i = 0; i < pct.size(); ++i)
        //{
        //    Point3D p = pct[i];
        //    h_pct.push_back(float4{ p.x, p.y, p.z, 0.0f });
        //}

        // Convert KDTree to array on the host
        size_t currentIndex = 0;
        flatten_KDTree(kdt.root_node_, h_array, currentIndex);

        // Transfer to device
        d_array = h_array;
        d_vAcc = h_vAcc;
        d_pct = h_pct;

        // Create texture object for d_vAcc (uint32_t -> int)
        cudaChannelFormatDesc channelDescVAcc = cudaCreateChannelDesc<int>();
        cudaResourceDesc resDescVAcc = {};
        resDescVAcc.resType = cudaResourceTypeArray;

        cudaArray_t d_vAccArray;
        cudaMallocArray(&d_vAccArray, &channelDescVAcc, d_vAcc.size());
        cudaMemcpyToArray(d_vAccArray, 0, 0, reinterpret_cast<int*>(d_vAcc.data().get()),
            d_vAcc.size() * sizeof(int), cudaMemcpyDeviceToDevice);

        resDescVAcc.res.array.array = d_vAccArray;

        cudaTextureDesc texDescVAcc = {};
        texDescVAcc.addressMode[0] = cudaAddressModeClamp;
        texDescVAcc.filterMode = cudaFilterModePoint;
        texDescVAcc.readMode = cudaReadModeElementType;
        texDescVAcc.normalizedCoords = 0;

        cudaCreateTextureObject(&texObjVAcc, &resDescVAcc, &texDescVAcc, nullptr);


        //// Create a channel description for float4
        //cudaChannelFormatDesc channelDescPct = cudaCreateChannelDesc<float4>();

        //// Allocate CUDA array for float4
        //cudaArray_t d_pctArray;
        //cudaMallocArray(&d_pctArray, &channelDescPct, d_pct.size());

        //// Copy data from d_pct to the CUDA array
        //cudaMemcpyToArray(d_pctArray, 0, 0, d_pct.data().get(),
        //    d_pct.size() * sizeof(float4), cudaMemcpyDeviceToDevice);

        //// Set up resource description
        //cudaResourceDesc resDescPct = {};
        //resDescPct.resType = cudaResourceTypeArray;
        //resDescPct.res.array.array = d_pctArray;

        //// Set up texture description
        //cudaTextureDesc texDescPct = {};
        //texDescPct.addressMode[0] = cudaAddressModeClamp; // Clamp for out-of-bounds access
        //texDescPct.filterMode = cudaFilterModePoint;      // No interpolation
        //texDescPct.readMode = cudaReadModeElementType;    // Read raw elements
        //texDescPct.normalizedCoords = 0;                 // Use unnormalized coordinates

        //// Create the texture object
        //cudaTextureObject_t texObjPct;
        //cudaCreateTextureObject(&texObjPct, &resDescPct, &texDescPct, nullptr);

    }

    FlattenedKDTree::~FlattenedKDTree()
    {
        cudaDestroyTextureObject(texObjArray);
        cudaDestroyTextureObject(texObjVAcc);
        cudaDestroyTextureObject(texObjPct);
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

    __device__ void FlattenedKDTree::find_nearest_neighbor(const Point3D query, size_t index, float& best_dist, size_t& best_idx, int depth) const
    {
        if (index >= d_array.size()) return;

        // Fetch node from texture object for device code
        //ArrayNode node = tex1Dfetch<ArrayNode>(texObjArray, index);
        ArrayNode node = d_array[index];
        
        if (node.is_leaf)
        {
            // Leaf node: Check all points in the leaf node
            size_t left = node.data.leaf.left;
            size_t right = node.data.leaf.right;
            for (size_t i = left; i <= right; i++)
            {
                // Fetch point index from texture object and then fetch point from texture
                int pointIdx_i = tex1Dfetch<int>(texObjVAcc, i);
                size_t pointIdx = reinterpret_cast<uint32_t&>(pointIdx_i);

                //float4 point_f4 = tex1Dfetch<float4>(texObjPct, static_cast<int>(pointIdx));
                //Point3D point{ point_f4.x, point_f4.y, point_f4.z };
                Point3D point = d_pct[i];

                // Compute the distance and update if it's the best
                float dist = distance_squared(query, point);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = pointIdx;
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