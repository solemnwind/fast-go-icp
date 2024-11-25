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
            ns{load_cloud(config.io.source, config.subsample, pcs)},
            nt{load_cloud(config.io.target, 1.0, pct)},
            registration{pct, nt, pcs, ns},
            max_iter(10), best_sse(M_INF), 
            best_translation(0.0f),
            mse_threshold(config.mse_threshold), // init *mean* squared error threshold 
            sse_threshold(ns* mse_threshold),    // init *sum* of squared error threshold
            stream_pool(32)
        {
            Logger(LogLevel::Info) << "Source points: " << ns << "\t"
                                   << "Target points: " << nt;

            // Set stack size to 16 KB to avoid stack overflow in recursion
            cudaDeviceSetLimit(cudaLimitStackSize, 16384);
            cudaCheckError("Set CUDA device stack size limit", false);
        };

        ~FastGoICP() {}

        void run();

    private:
        icp::Config config;

        // Data
        PointCloud pcs;  // source point cloud
        PointCloud pct;  // target point cloud
        size_t ns, nt; // number of source/target points

        // Registration object for ICP and error computing
        Registration registration;

        // Runtime variables
        size_t max_iter;
        float best_sse;
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

        // CUDA stream pool
        StreamPool stream_pool;

    private:
        using ResultBnBR3 = std::tuple<float, glm::vec3>;

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
        ResultBnBR3 branch_and_bound_R3(RotNode &rnode, bool fix_rot);
    };
}

#endif // FGOICP_HPP