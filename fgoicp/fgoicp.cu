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
            ResultBnBR3 res = branch_and_bound_R3(rnode);
            if (res.ub < best_sse)
            {
                best_sse = res.ub;
                best_rotation = rnode.q;
                best_translation = res.best_translation;
                Logger(LogLevel::Debug) << "New best error: " << best_sse << "\n"
                                        << "\tRotation:\n" << best_rotation.R
                                        << "\tTranslation: " << best_translation;
            }
            if (best_sse <= sse_threshold)
            {
                break;
            }
            if (res.lb > best_sse) { continue; }


            // Spawn children RotNodes

        }
        return best_sse;
    }

    FastGoICP::ResultBnBR3 FastGoICP::branch_and_bound_R3(RotNode &rnode)
    {
        // TODO:  Need target point cloud stats
        // Assume the the target point cloud is within AABB [-2.0, -2.0, -1.0, 2.0, 2.0, 1.0]
        float xmin = -1.0;
        float ymin = -1.0;
        float zmin = -1.0;
        float xmax = 1.0;
        float ymax = 1.0;
        float zmax = 1.0;

        // Everything for Rotation Lower/Upper Bound
        struct 
        {
            float best_error;
            glm::vec3 best_translation{ 0.0f };
            std::vector<float> trans_lb;    // TODO: use std::array<float, 32>
            std::vector<float> trans_ub;
            size_t idx_min;
        } rot_lb, rot_ub;
        rot_lb.best_error = this->best_sse;
        rot_ub.best_error = this->best_sse;

        // Initialize queue for TransNodes
        std::priority_queue<TransNode> tcandidates;
        {
            constexpr float step = 1.0f / 2.0f;
            constexpr float span = step / 2.0f;
            for (float x = xmin + span; x < xmax; x += step)
            {
                for (float y = ymin + span; y < ymax; y += step)
                {
                    for (float z = zmin + span; z < zmax; z += step)
                    {
                        TransNode tnode = TransNode(x, y, z, span, 0.0f, rnode.ub);
                        tcandidates.push(std::move(tnode));
                    }
                }
            }
        }

        while (!tcandidates.empty())
        {
            std::vector<TransNode> tnodes;

            // Get a batch
            while (!tcandidates.empty() && tnodes.size() < 64)
            {
                auto tnode = tcandidates.top();
                tcandidates.pop();
                tnodes.push_back(std::move(tnode));
            }

            // Compute lower/upper bounds
            std::tie(rot_lb.trans_lb, 
                     rot_lb.trans_ub, 
                     rot_ub.trans_lb, 
                     rot_ub.trans_ub) = registration.compute_sse_error(rnode.q, tnodes, stream_pool);

            // *Fix rotation* to compute rotation *lower bound*
            // Get min upper bound of this batch to update best SSE
            rot_ub.idx_min = std::distance(std::begin(rot_ub.trans_ub),
                                           std::min_element(std::begin(rot_ub.trans_ub),
                                                            std::end(rot_ub.trans_ub)));
            if (rot_ub.trans_ub[rot_ub.idx_min] < rot_ub.best_error)
            {
                rot_ub.best_error = rot_ub.trans_ub[rot_ub.idx_min];
                rot_ub.best_translation = tnodes[rot_ub.idx_min].t;
            }

            // Examine translation lower bounds
            for (size_t i = 0; i < tnodes.size(); ++i)
            {
                // Eliminate those with lower bound >= best SSE
                if (rot_ub.trans_lb[i] >= rot_ub.best_error ||
                    rot_lb.trans_lb[i] >= rot_lb.best_error) 
                { continue; }

                TransNode& tnode = tnodes[i];
                // Stop if the span is small enough
                if (tnode.span < 0.05f) { continue; }  // TODO: use config threshold

                float span = tnode.span / 2.0f;
                // Spawn 8 children
                for (char j = 0; j < 8; ++j)
                {
                    TransNode child_tnode(
                        tnode.t.x - span + (j >> 0 & 1) * tnode.span,
                        tnode.t.y - span + (j >> 1 & 1) * tnode.span,
                        tnode.t.z - span + (j >> 2 & 1) * tnode.span,
                        span, rot_lb.trans_lb[i], rot_ub.trans_lb[i]
                    );
                    tcandidates.push(std::move(child_tnode));
                }
            }

        }

        Logger(LogLevel::Debug) << "Inner BnB finished, best error: " << rot_ub.best_error;

        return { rot_lb.best_error, rot_ub.best_error, rot_ub.best_translation };
    }
}