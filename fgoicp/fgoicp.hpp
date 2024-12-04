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
        FastGoICP(std::vector<glm::vec3> _pct, std::vector<glm::vec3> _pcs, float _mse_threshold) : 
            pcs(_pcs), pct(_pct), ns{pcs.size()}, nt{pct.size()},
            offset_pcs(center_point_cloud(pcs)),
            offset_pct(center_point_cloud(pct)),
            scaling_factor(scale_point_clouds(pct, pcs)),
            target_bounds(get_point_cloud_ranges(pct)),
            registration{pct, pcs, target_bounds, 0.03},
            max_iter(10), best_sse(M_INF), 
            best_rotation(1.0f), best_translation(0.0f),
            mse_threshold(_mse_threshold), // init *mean* squared error threshold 
            sse_threshold(ns * mse_threshold),    // init *sum* of squared error threshold
            stream_pool(32)
        {
            
        };

        ~FastGoICP() {}

        using Result_t = std::tuple<glm::mat3, glm::vec3>;
        Result_t run();

        // Interfaces for visualization
        float get_best_error() const { return best_sse; }

        Result_t get_best_transform() const
        {
            return { best_rotation, best_translation };
        }

        Result_t get_last_transform() const
        {
            return { last_rotation, last_translation };
        }

    private:
        // Data
        PointCloud pcs;  // source point cloud
        PointCloud pct;  // target point cloud
        size_t ns, nt; // number of source/target points

        // Preprocess
        glm::vec3 offset_pcs;
        glm::vec3 offset_pct;
        float scaling_factor;
        std::array<std::pair<float, float>, 3> target_bounds;

        // Registration object for ICP and error computing
        Registration registration;

        // Runtime variables
        size_t max_iter;
        float best_sse;
        glm::mat3 best_rotation;
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

        glm::mat3 last_rotation{ 1.0f };
        glm::vec3 last_translation{ 0.0f };

    private:
        glm::vec3 center_point_cloud(PointCloud& pc);
        float scale_point_clouds(PointCloud& pct, PointCloud& pcs);
        std::array<std::pair<float, float>, 3> get_point_cloud_ranges(PointCloud& pc);

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