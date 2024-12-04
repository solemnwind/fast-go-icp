#include "registration.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector_types.h>
#include <texture_types.h>
#include <utility>

namespace icp
{
    __global__ void kernComputeClosestError(int ns, int nt, glm::mat3 R, glm::vec3 t, const Point3D *d_pcs, const Point3D* d_pct, float* d_errors)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= ns) { return; }

        Point3D source_point = d_pcs[index];
        Point3D query_point = R * source_point + t;

        float distance_squared = brute_force_find_nearest_neighbor(query_point, d_pct, nt);

        d_errors[index] = distance_squared;
    }

    __global__ void kernComputeBounds(int N, RotNode rnode, TransNode tnode, bool fix_rot, const Point3D* d_pcs, const NearestNeighborLUT* d_lut, float* d_rot_ub_trans_ub, float* d_rot_ub_trans_lb)
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

        float distance_squared = d_lut->search(float3{ query_point.x, query_point.y, query_point.z });

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
            ns, nt, R, t,
            d_pcs,
            d_pct,
            dev_errors);
        cudaDeviceSynchronize();
        cudaCheckError("kernComputeClosestError");

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
        std::vector<float> sse_upper_bounds(num_transforms);
        std::vector<float> sse_lower_bounds(num_transforms);

        // Allocate memory on the device for the errors for each (R, t) pair
        float* d_upper_bounds;
        float* d_lower_bounds;
        cudaMalloc((void**)&d_upper_bounds, sizeof(float) * ns * num_transforms);
        cudaMalloc((void**)&d_lower_bounds, sizeof(float) * ns * num_transforms);

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
                d_pcs,
                d_nnlut,
                d_upper_bounds + i * ns,
                d_lower_bounds + i * ns);
        }

        // Reduce the lower/upper bounds for each pair
        for (size_t i = 0; i < num_transforms; ++i) {
            // Thrust reduce launching parameters
            auto thrust_policy = thrust::cuda::par.on(stream_pool.getStream(i));

            sse_upper_bounds[i] = thrust::reduce(
                thrust_policy,
                d_upper_bounds + i * ns,
                d_upper_bounds + (i + 1) * ns,
                0.0f,
                thrust::plus<float>()
            );

            sse_lower_bounds[i] = thrust::reduce(
                thrust_policy,
                d_lower_bounds + i * ns,
                d_lower_bounds + (i + 1) * ns,
                0.0f,
                thrust::plus<float>()
            );
        }

        cudaDeviceSynchronize();

        // Free the device memory
        cudaFree(d_upper_bounds);
        cudaFree(d_lower_bounds);

        return { sse_lower_bounds, sse_upper_bounds };
    }

    __device__ float distance_squared(const Point3D p1, const Point3D p2)
    {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ float brute_force_find_nearest_neighbor(const Point3D query, const Point3D* d_pct, size_t nt)
    {
        float best_dist = M_INF;
        for (size_t i = 0; i < nt; ++i)
        {
            float dist_sq = distance_squared(query, d_pct[i]);
            if (dist_sq < best_dist)
            {
                best_dist = dist_sq;
            }
        }
        return best_dist;
    }

    //============================================
    //                   LUT
    //============================================

    NearestNeighborLUT::NearestNeighborLUT(float resolution, const std::array<std::pair<float, float>, 3> _target_bounds, const PointCloud& pc) : 
        resolution(resolution), 
        target_bounds(_target_bounds),
        d_cudaArray(nullptr), 
        d_lutData(nullptr),
        texObj(0)
    {
        dims = make_int3(ceil((target_bounds[0].second - target_bounds[0].first) / resolution),
                         ceil((target_bounds[1].second - target_bounds[1].first) / resolution),
                         ceil((target_bounds[2].second - target_bounds[2].first) / resolution));
        // Calculate scale factors and offsets based on target_bounds
        scale = 1.0f / resolution;
        offset.x = -target_bounds[0].first;
        offset.y = -target_bounds[1].first;
        offset.z = -target_bounds[2].first;

        this->build(pc);
    }

    NearestNeighborLUT::~NearestNeighborLUT()
    {
        cleanupCudaTexture();
    }

    void NearestNeighborLUT::initializeCudaTexture()
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaMalloc3DArray(&d_cudaArray, &channelDesc, make_cudaExtent(dims.x, dims.y, dims.z));

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = d_cudaArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    }

    void NearestNeighborLUT::cleanupCudaTexture()
    {
        if (texObj)
        {
            cudaDestroyTextureObject(texObj);
            texObj = 0;
        }
        if (d_cudaArray)
        {
            cudaFreeArray(d_cudaArray);
            d_cudaArray = nullptr;
        }
        if (d_lutData)
        {
            cudaFree(d_lutData);
            d_lutData = nullptr;
        }
    }

    __device__ __host__ inline float distance_squared(const float3& u, const float3& v)
    {
        float dx = u.x - v.x;
        float dy = u.y - v.y;
        float dz = u.z - v.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __global__ void buildLUTKernel(float* lutData, int3 dims, float resolution, const float3* points, int numPoints)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x >= dims.x || y >= dims.y || z >= dims.z) return;

        float3 cellCenter = make_float3(x * resolution, y * resolution, z * resolution);
        float minDist = FLT_MAX;

        for (int i = 0; i < numPoints; ++i)
        {
            float3 point = make_float3(points[i].x, points[i].y, points[i].z);
            float dist_sq = distance_squared(cellCenter, point);
            minDist = minDist < dist_sq ? minDist : dist_sq;
        }

        int index = (z * dims.y + y) * dims.x + x;
        lutData[index] = minDist;
    }

    void NearestNeighborLUT::build(const PointCloud& pc)
    {
        cleanupCudaTexture();
        initializeCudaTexture();

        int numPoints = pc.size();
        thrust::host_vector<float3> h_points(numPoints);



        // Transform points into LUT space
        for (int i = 0; i < numPoints; ++i)
        {
            h_points[i] = make_float3(
                (pc[i].x + offset.x),
                (pc[i].y + offset.y),
                (pc[i].z + offset.z)
            );
        }

        thrust::device_vector<float3> d_points = h_points;

        cudaMalloc(&d_lutData, dims.x * dims.y * dims.z * sizeof(float));

        dim3 blockSize(8, 8, 8);
        dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x,
            (dims.y + blockSize.y - 1) / blockSize.y,
            (dims.z + blockSize.z - 1) / blockSize.z);

        buildLUTKernel <<<gridSize, blockSize>>> (d_lutData, dims, resolution, thrust::raw_pointer_cast(d_points.data()), numPoints);

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr(d_lutData, dims.x * sizeof(float), dims.x, dims.y);
        copyParams.dstArray = d_cudaArray;
        copyParams.extent = make_cudaExtent(dims.x, dims.y, dims.z);
        copyParams.kind = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParams);
    }

    __device__ float NearestNeighborLUT::search(const float3 query) const
    {
        // Map query point into LUT space using target_bounds
        float x = (query.x + offset.x) * scale;
        float y = (query.y + offset.y) * scale;
        float z = (query.z + offset.z) * scale;

        return tex3D<float>(texObj, x, y, z);
    }
}