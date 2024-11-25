#include "icp3d.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <Eigen/Dense>

namespace icp
{
    __global__ void kernFindNearestNeighbor(int nt, int ns, const Point3D* d_pct, const Point3D* d_pcs, Point3D* d_corrs)
    {
	    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= ns) { return; }

        float dist_min = M_INF;
        Point3D corr(0.0f);
        for (int j = 0; j < nt; ++j) 
        {
            float dist = glm::distance(d_pcs[index], d_pct[j]);
            if (dist_min > dist) 
            {
                dist_min = dist;
                corr = d_pct[j];
            }
        }
        d_corrs[index] = corr;
    }

    __global__ void kernRotateTranslateInplace(int N, glm::mat3 R, glm::vec3 t, Point3D* d_pc)
    {
	    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        d_pc[index] = R * d_pc[index] + t;
    }

    __global__ void kernCentralize(int numDataPoints, Point3D centroid, Point3D* d_in, Point3D* d_out) 
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= numDataPoints) return;

        d_out[index] = d_in[index] - centroid;
    }

    __global__ void kernOuterProduct(int numDataPoints, glm::vec3* A, glm::vec3* B, glm::mat3* out) 
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= numDataPoints) return;

        out[index] = glm::outerProduct(A[index], B[index]);
    }


    IterativeClosestPoint3D::IterativeClosestPoint3D(const Registration &reg, const PointCloud &pct, const PointCloud &pcs, size_t max_iter, float sse_threshold, glm::mat3 R, glm::vec3 t) :
        reg(reg), nt(pct.size()), ns(pcs.size()), R(R), t(t), sse(M_INF),
        max_iter(max_iter), sse_threshold(sse_threshold)
    {
        cudaMalloc((void**)&d_pct_buffer, sizeof(Point3D) * nt);
        cudaMalloc((void**)&d_pcs_buffer, sizeof(Point3D) * ns);
        cudaMalloc((void**)&d_corrs_buffer, sizeof(Point3D) * ns);
        cudaMalloc((void**)&d_pcs_centered_buffer, sizeof(Point3D) * ns);
        cudaMalloc((void**)&d_corrs_centered_buffer, sizeof(Point3D) * ns);
        cudaMalloc((void**)&d_mat_buffer, sizeof(glm::mat3) * ns);

        cudaMemcpy(d_pct_buffer, pct.data(), sizeof(Point3D) * nt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pcs_buffer, pcs.data(), sizeof(Point3D) * ns, cudaMemcpyHostToDevice);

        const size_t block_size = 256;
        const dim3 threads_per_block(block_size);
        const dim3 blocks_per_grid((ns + block_size - 1) / block_size);
        kernRotateTranslateInplace <<<blocks_per_grid, threads_per_block>>> (ns, R, t, d_pcs_buffer);
        cudaCheckError("kernRotateTranslateInplace");

        size_t iter = 0;
        while (iter++ < max_iter && sse > sse_threshold)
        {
            auto [R_, t_] = procrustes();
            kernRotateTranslateInplace <<<blocks_per_grid, threads_per_block>>> (ns, R_, t_, d_pcs_buffer);
            R = R_ * R;
            t = R * t + t_;
            sse = reg.compute_sse_error(R, t);
        }
    }

    IterativeClosestPoint3D::~IterativeClosestPoint3D()
    {
        cudaFree(d_pct_buffer);
        cudaFree(d_pcs_buffer);
        cudaFree(d_corrs_buffer);
        cudaFree(d_pcs_centered_buffer);
        cudaFree(d_corrs_centered_buffer);
        cudaFree(d_mat_buffer);
    }

    IterativeClosestPoint3D::Result_t IterativeClosestPoint3D::get_result() const
    {
        return { this->sse, this->R, this->t };
    }

    glm::mat3 closest_orthogonal_approximation(glm::mat3 ABt)
    {
        Eigen::Matrix3d matrix;
        // Transpose b/c. of different major
        matrix << ABt[0][0], ABt[1][0], ABt[2][0],
                  ABt[0][1], ABt[1][1], ABt[2][1],
                  ABt[0][2], ABt[1][2], ABt[2][2];

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Vector3d S = svd.singularValues();
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        Eigen::Matrix3d VU_T = V * U.transpose();

        // Compute determinant of VU^T
        double det_VU_T = VU_T.determinant();

        // Create diagonal matrix diag(1, 1, det(VU^T))
        Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
        D(2, 2) = det_VU_T;

        // Compute R = V * D * U^T
        Eigen::Matrix3d R = V * D * U.transpose(); 
        
        return glm::mat3{R(0, 0), R(1, 0), R(2, 0),
                         R(0, 1), R(1, 1), R(2, 1),
                         R(0, 2), R(1, 2), R(2, 2)};
    }

    IterativeClosestPoint3D::RigidMotion_t IterativeClosestPoint3D::procrustes()
    {
        constexpr size_t block_size = 256;
        const dim3 threads_per_block(block_size);
        const dim3 blocks_per_grid((ns + block_size - 1) / block_size); 
        
        kernFindNearestNeighbor <<<blocks_per_grid, threads_per_block>>> (nt, ns, d_pct_buffer, d_pcs_buffer, d_corrs_buffer);
        cudaDeviceSynchronize();

        // Could use stream to speed up a bit
        thrust::device_ptr<Point3D> d_thrust_pcs_buffer(d_pcs_buffer);
        thrust::device_ptr<Point3D> d_thrust_corrs_buffer(d_corrs_buffer);
        Point3D src_centroid = thrust::reduce(d_thrust_pcs_buffer, d_thrust_pcs_buffer + ns);
        Point3D cor_centroid = thrust::reduce(d_thrust_corrs_buffer, d_thrust_corrs_buffer + ns);

        src_centroid /= static_cast<float>(ns);
        cor_centroid /= static_cast<float>(ns);

        kernCentralize <<<blocks_per_grid, threads_per_block>>> (ns, src_centroid, d_pcs_buffer, d_pcs_centered_buffer);
        kernCentralize <<<blocks_per_grid, threads_per_block>>> (ns, cor_centroid, d_corrs_buffer, d_corrs_centered_buffer);
        cudaDeviceSynchronize();

        kernOuterProduct <<<blocks_per_grid, threads_per_block>>> (ns, d_pcs_centered_buffer, d_corrs_centered_buffer, d_mat_buffer);
        cudaDeviceSynchronize();

        thrust::device_ptr<glm::mat3> d_thrust_mat_buffer(d_mat_buffer);
        glm::mat3 ABt = thrust::reduce(d_thrust_mat_buffer, d_thrust_mat_buffer + ns);

        glm::mat3 R_ = closest_orthogonal_approximation(ABt);
        glm::vec3 t_ = cor_centroid - R_ * src_centroid;

        return { R_, t_ };
    }
}
