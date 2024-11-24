#include "fgoicp.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <queue>
#include <tuple>

namespace icp
{
    void FastGoICP::run()
    {
        best_sse = registration.compute_sse_error(glm::mat3(1.0f), glm::vec3(0.0f));
        Logger(LogLevel::Info) << "Initial best error: " << best_sse;

        branch_and_bound_SO3();
        Logger(LogLevel::Info) << "Searching over...";
    }

    float FastGoICP::branch_and_bound_SO3()
    {
        // Initialize Rotation Nodes
        std::queue<RotNode> rcandidates;
        {
            constexpr float N = 8.0f;
            constexpr float step = 2.0f / N;
            constexpr float span = 1.0f / N;
            constexpr float start = -1.0f + span;
            for (float x = start; x < 1.0f; x += step)
            {
                for (float y = start; y < 1.0f; y += step)
                {
                    for (float z = start; z < 1.0f; z += step)
                    {
                        RotNode rnode = RotNode(x, y, z, span, 0.0f, M_INF);
                        if (rnode.overlaps_SO3() && rnode.q.in_SO3()) { rcandidates.push(std::move(rnode)); }
                    }
                }
            }
        }

        while (!rcandidates.empty())
        {
            RotNode rnode = rcandidates.front();
            rcandidates.pop();
                    
            // BnB in R3 
            auto [ub, best_t] = branch_and_bound_R3(rnode);
            if (ub < best_sse)
            {
                best_sse = ub;
                best_rotation = rnode.q;
                best_translation = best_t;
                Logger(LogLevel::Debug) << "New best error: " << best_sse << "\n"
                                        << "\tRotation:\n" << best_rotation.R
                                        << "\tTranslation: " << best_t;
            }
            if (best_sse <= sse_threshold)
            {
                break;
            }


            // Spawn children RotNodes

        }
        return best_sse;
    }

    FastGoICP::ResultBnBR3 FastGoICP::branch_and_bound_R3(RotNode &rnode)
    {
        // TODO:  Need target point cloud stats
        // Assume the the target point cloud is within AABB [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        float xmin = -1.0;
        float ymin = -1.0;
        float zmin = -1.0;
        float xmax = 1.0;
        float ymax = 1.0;
        float zmax = 1.0;

        // Everything for Rotation Upper Bound

        float best_error = this->best_sse;
        glm::vec3 best_t{ 0.0f };

        size_t count = 0;

        // Initialize queue for TransNodes
        std::priority_queue<TransNode> tcandidates;

        TransNode tnode = TransNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, rnode.ub);
        tcandidates.push(std::move(tnode));

        while (!tcandidates.empty())
        {
            std::vector<TransNode> tnodes;

            if (best_error - tcandidates.top().lb < sse_threshold) { break; }
            // Get a batch
            while (!tcandidates.empty() && tnodes.size() < 16)
            {
                auto tnode = tcandidates.top();
                tcandidates.pop();
                if (tnode.lb < best_error)
                {
                    tnodes.push_back(std::move(tnode));
                }
            }

            count += tcandidates.size();

            // Compute lower/upper bounds
            auto [lb, ub] = registration.compute_sse_error(rnode.q, tnodes, stream_pool);

            // *Fix rotation* to compute rotation *lower bound*
            // Get min upper bound of this batch to update best SSE
            size_t idx_min = std::distance(std::begin(ub), std::min_element(std::begin(ub), std::end(ub)));
            if (ub[idx_min] < best_error)
            {
                best_error = ub[idx_min];
                best_t = tnodes[idx_min].t;
            }

            // Examine translation lower bounds
            for (size_t i = 0; i < tnodes.size(); ++i)
            {
                // Eliminate those with lower bound >= best SSE
                if (lb[i] >= best_error) { continue; }

                TransNode& tnode = tnodes[i];
                // Stop if the span is small enough
                if (tnode.span < 0.1f) { continue; }  // TODO: use config threshold

                float span = tnode.span / 2.0f;
                // Spawn 8 children
                for (char j = 0; j < 8; ++j)
                {
                    TransNode child_tnode(
                        tnode.t.x - span + (j >> 0 & 1) * tnode.span,
                        tnode.t.y - span + (j >> 1 & 1) * tnode.span,
                        tnode.t.z - span + (j >> 2 & 1) * tnode.span,
                        span, lb[i], ub[i]
                    );
                    tcandidates.push(std::move(child_tnode));
                }
            }

        }

        Logger() << count << " TransNodes searched. Inner BnB finished, best error: " << best_error;

        return { best_error, best_t };
    }
}