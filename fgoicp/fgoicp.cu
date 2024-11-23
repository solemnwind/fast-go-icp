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
        best_sse = registration.compute_sse_error(glm::mat3(1.0f), glm::vec3(0.0f));
        Logger(LogLevel::Info) << "Initial best error: " << best_sse;

        branch_and_bound_SO3();
    }

    float FastGoICP::branch_and_bound_SO3()
    {
        // Initialize Rotation Nodes
        std::queue<RotNode> rcandidates;
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
                        RotNode rnode = RotNode(x, y, z, span, M_INF, 0.0f);
                        if (rnode.overlaps_SO3()) { rcandidates.push(std::move(rnode)); }
                    }
                }
            }
        }

        while (!rcandidates.empty())
        {
            RotNode rnode = rcandidates.front();
            rcandidates.pop();
                    
            // BnB in R3 
            ResultBnBR3 res = branch_and_bound_R3(rnode.q);
            if (res.error < best_sse)
            {
                best_sse = res.error;
                best_rotation = rnode.q;
                best_translation = res.translation;
                Logger(LogLevel::Debug) << "New best error: " << best_sse << "\n"
                                        << "\tRotation:\n" << best_rotation.R
                                        << "\tTranslation: " << best_translation;
            }
            //if (res.error < sse_threshold)
            //{
            //    return res.error;
            //}
        }
        return 0.0f;
    }

    FastGoICP::ResultBnBR3 FastGoICP::branch_and_bound_R3(Rotation q)
    {
        // TODO:  Need target point cloud stats
        // Assume the the target point cloud is within AABB [-2.0, -2.0, -1.0, 2.0, 2.0, 1.0]
        float xmin = -1.0;
        float ymin = -1.0;
        float zmin = -1.0;
        float xmax = 1.0;
        float ymax = 1.0;
        float zmax = 1.0;

        // Initialize
        std::queue<TransNode> tcandidates;    // Consider using priority queue
        {
            float step = 1.0f / 4.0f;
            float span = 1.0f / 8.0f;
            for (float x = xmin + span; x < xmax; x += step)
            {
                for (float y = ymin + span; y < ymax; y += step)
                {
                    for (float z = zmin + span; z < zmax; z += step)
                    {
                        TransNode tnode = TransNode(x, y, z, span, M_INF, 0.0f);
                        tcandidates.push(std::move(tnode));
                    }
                }
            }
        }

        float best_sse_ = M_INF;
        glm::vec3 best_translation_(0.0f);

        while (!tcandidates.empty())
        {
            std::vector<glm::mat3> Rs;
            std::vector<glm::vec3> ts;
            std::vector<TransNode> tnodes;

            for (int i = 0; i < 64; ++i)
            {
                if (tcandidates.empty()) { break; }
                auto tnode = tcandidates.front();
                tcandidates.pop();
                Rs.push_back(q.R);
                ts.push_back(tnode.t);
                tnodes.push_back(std::move(tnode));
            }

            auto sses = registration.compute_sse_error(Rs, ts, stream_pool);
            size_t index = std::distance(std::begin(sses), std::min_element(std::begin(sses), std::end(sses)));
            if (sses[index] < best_sse_)
            {
                best_sse_ = sses[index];
                best_translation_ = tnodes[index].t;
            }

        }

        return { best_sse_, best_translation_ };
    }
}