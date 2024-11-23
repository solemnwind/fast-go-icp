#include "fgoicp.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <queue>

namespace icp
{
    void FastGoICP::run()
    {
        best_error = registration.compute_sse_error(glm::mat3(1.0f), glm::vec3(0.0f));
        Logger(LogLevel::Info) << "Initial best error: " << best_error;
    }

    float FastGoICP::branch_and_bound_SO3()
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

    FastGoICP::ResultBnBR3 FastGoICP::branch_and_bound_R3(Rotation q)
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
}