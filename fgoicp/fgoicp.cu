#include "fgoicp.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <queue>
#include <tuple>
#include "icp3d.hpp"

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
        std::priority_queue<RotNode> rcandidates;
        RotNode rnode = RotNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, this->best_sse);
        rcandidates.push(std::move(rnode));

#define GROUND_TRUTH
#ifdef GROUND_TRUTH
        RotNode gt_rnode = RotNode(0.0625f, /*-1.0f / sqrt(2.0f)*/ -0.85f, 0.0625f, 0.0625f, 0.0f, M_INF);
        //RotNode gt_rnode = RotNode(0.0f, -1.0f / sqrt(2.0f), 0.0f, 0.0625f, 0.0f, M_INF);
        Logger() << "Ground Truth Rot:\n" << gt_rnode.q.R;
        auto [cub, t] = branch_and_bound_R3(gt_rnode, true);
        auto [clb, _] = branch_and_bound_R3(gt_rnode, false);
        Logger() << "Correct, ub: " << cub << " lb: " << clb << " t:\n\t" << t;

        IterativeClosestPoint3D icp3d(registration, pct, pcs, 2000, sse_threshold, gt_rnode.q.R, t);
        return best_sse;
#endif

        while (!rcandidates.empty())
        {
            RotNode rnode = rcandidates.top();
            rcandidates.pop();

            if (best_sse - rnode.lb <= sse_threshold)
            {
                break;
            }

            // Spawn children RotNodes
            float span = rnode.span / 2.0f;
            for (char j = 0; j < 8; ++j)
            {
                if (span < 0.02f) { continue; }
                RotNode child_rnode(
                    rnode.q.x - span + (j >> 0 & 1) * rnode.span,
                    rnode.q.y - span + (j >> 1 & 1) * rnode.span,
                    rnode.q.z - span + (j >> 2 & 1) * rnode.span,
                    span, rnode.lb, rnode.ub
                );

                if (!child_rnode.overlaps_SO3()) { continue; }
                if (!child_rnode.q.in_SO3()) 
                { 
                    rcandidates.push(std::move(child_rnode)); 
                    continue; 
                }

                Logger() << "Rotation:\t" << glm::vec3{ child_rnode.q.x, child_rnode.q.y, child_rnode.q.z }
                    << "\tspan: " << child_rnode.span << "\tr: " << child_rnode.q.r;
                // BnB in R3 
                auto [ub, best_t] = branch_and_bound_R3(child_rnode, true);
                Logger() << "ub: " << ub;

                if (ub < best_sse)
                {
                    // TODO: ICP here
                    best_sse = ub;
                    best_rotation = child_rnode.q;
                    best_translation = best_t;
                    Logger(LogLevel::Debug) << "New best error: " << best_sse << "\n"
                        << "\tRotation:\n" << best_rotation.R << "\n"
                        << "\tTranslation: " << best_t;
                }

                auto [lb, _] = branch_and_bound_R3(child_rnode, false);
                Logger() << "lb: " << lb;

                if (lb >= best_sse) { continue; }
                child_rnode.lb = lb;
                child_rnode.ub = ub;

                rcandidates.push(std::move(child_rnode));
            }
        }
        return best_sse;
    }

    FastGoICP::ResultBnBR3 FastGoICP::branch_and_bound_R3(RotNode &rnode, bool fix_rot)
    {
        float best_error = this->best_sse;
        glm::vec3 best_t{ 0.0f };

        size_t count = 0;

        // Initialize queue for TransNodes
        std::priority_queue<TransNode> tcandidates;

        TransNode init_tnode = TransNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, rnode.ub);
        tcandidates.push(std::move(init_tnode));

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

            count += tnodes.size();

            // Compute lower/upper bounds
            auto [lb, ub] = registration.compute_sse_error(rnode, tnodes, fix_rot, stream_pool);

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
                if (tnode.span < 0.2f) { continue; }  // TODO: use config threshold

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

        Logger() << count << " TransNodes searched. Inner BnB finished";

        return { best_error, best_t };
    }
}