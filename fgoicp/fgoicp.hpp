#ifndef FGOICP_HPP
#define FGOICP_HPP

#include "common.hpp"
#include "registration.hpp"
#include <iostream>

namespace icp
{
    class FastGoICP
    {
    public:
        FastGoICP(std::string config_path) : 
            config{config_path}, pcs(), pct(),
            ns{load_cloud_ply(config.io.source, config.subsample, pcs)},
            nt{load_cloud_ply(config.io.target, 1.0, pct)},
            registration{pct, nt, pcs, ns},
            max_iter(10), best_error(M_INF), 
            best_translation(0.0f),
            mse_threshold(1E-3f),               // init *mean* squared error threshold 
            sse_threshold(ns* mse_threshold)    // init *sum* of squared error threshold
        {
            Logger(LogLevel::Info) << "Source points: " << ns << "\t"
                                   << "Target points: " << nt;

            // Set stack size to 16 KB to avoid stack overflow in recursion
            if (cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, 16384); err != cudaSuccess) 
            {
                Logger(LogLevel::Warning) << "Error setting stack size: " << cudaGetErrorString(err);
            }
        };

        ~FastGoICP() {}

        void run();

    private:
        icp::Config config;

        // Data
        PointCloud pcs;  // source point cloud
        PointCloud pct;  // target point cloud
        size_t ns, nt; // number of source/target points

        Registration registration;

        // Runtime variables
        size_t max_iter;
        float best_error;
        Rotation best_rotation;
        glm::vec3 best_translation;

        // MSE threshold depends on the source point cloud stats.
        // If we normalize the source point cloud into a standard cube,
        // The MSE threshold can be specified without considering 
        // the point cloud stats.
        float mse_threshold;
        // SSE threshold is the summed error threshold,
        // the registration is considered converged if SSE threshold is reached.
        // If no trimming, sse_threshold = ns * mse_threshold
        float sse_threshold;

    private:
        struct ResultBnBR3
        {
            float error;
            glm::vec3 translation;
        };

        /**
         * @brief Perform branch-and-bound algorithm in Rotation Space SO(3)
         *
         * @return float
         */
        float branch_and_bound_SO3();

        /**
         * @brief Perform branch-and-bound algorithm in Translation Space R(3)
         *
         * @param
         * @return ResultBnBR3
         */
        ResultBnBR3 branch_and_bound_R3(Rotation q);
    };
}

#endif // FGOICP_HPP