#include "fgoicp.hpp"
#include "icp3d.hpp"
#include <iostream>
#include <queue>
#include <tuple>
#include <omp.h>

namespace icp
{
    FastGoICP::Result_t FastGoICP::run()
    {
        IterativeClosestPoint3D icp3d(registration, pct, pcs, 100, 0.05, glm::mat3(1.0f), glm::vec3(0.0f));
        auto [icp_sse, icp_R, icp_t] = icp3d.run();
        best_sse = icp_sse;
        Logger(LogLevel::Info) << "Initial ICP best error: " << icp_sse
                               << "\n\tRotation:\n" << icp_R
                               << "\n\tTranslation: " << icp_t;

        branch_and_bound_SO3();

        // Refine the best tranform
        IterativeClosestPoint3D icp3d_best(registration, pct, pcs, 100, 0.0005, best_rotation, best_translation);
        std::tie(best_sse, best_rotation, best_translation) = icp3d_best.run();

        Logger(LogLevel::Info) << "Searching over! Best Error: " << best_sse
                               << "\n\tRotation:\n" << best_rotation
                               << "\n\tTranslation: " << restore_translation(best_rotation, best_translation);

        return { best_rotation, restore_translation(best_rotation, best_translation)};
    }

    float FastGoICP::branch_and_bound_SO3()
    {
        // Initialize Rotation Nodes
        std::priority_queue<RotNode> rcandidates;
        RotNode rnode = RotNode(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, this->best_sse);
        rcandidates.push(std::move(rnode));

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
                if (span < 0.05f) { continue; }
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

                // BnB in R3 
                auto [ub, best_t] = branch_and_bound_R3(child_rnode, true);

                last_rotation = child_rnode.q.R;
                last_translation = best_t;

                if (ub < best_sse * 1.8)
                {
                    IterativeClosestPoint3D icp3d(registration, pct, pcs, 100, 0.005, child_rnode.q.R, best_t);
                    auto [icp_sse, icp_R, icp_t] = icp3d.run();

                    if (icp_sse < best_sse)
                    {
                        best_sse = icp_sse;
                        best_rotation = icp_R;
                        best_translation = icp_t;
                    } 
                    Logger(LogLevel::Debug) << "New best error: " << best_sse
                        << "\n\tRotation:\n" << best_rotation
                        << "\n\tTranslation: " << restore_translation(best_rotation, best_translation);
                }

                auto [lb, _] = branch_and_bound_R3(child_rnode, false);

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
        float best_ub = M_INF;

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
            while (!tcandidates.empty() && tnodes.size() < 32)
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
            best_ub = best_ub < ub[idx_min] ? best_ub : ub[idx_min];
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

        return { best_ub, best_t };
    }

    glm::vec3 FastGoICP::center_point_cloud(PointCloud& pc) {
        glm::vec3 centroid(0.0f);

        // Calculate centroid using OpenMP
        #pragma omp parallel for reduction(+:centroid)
        for (size_t i = 0; i < pc.size(); ++i) 
        {
            centroid += pc[i];
        }
        centroid /= static_cast<float>(pc.size());

        // Translate points to center them around the origin
        #pragma omp parallel for
        for (size_t i = 0; i < pc.size(); ++i) 
        {
            pc[i] -= centroid;
        }

        return -centroid;
    }

    float get_scaling_factor(PointCloud& pc)
    {
        float max_abs = std::numeric_limits<float>::lowest();

        #pragma omp parallel
        {
            float local_max_abs = std::numeric_limits<float>::lowest();

            #pragma omp for nowait
            for (size_t i = 0; i < pc.size(); ++i)
            {
                const auto& p = pc[i];
                local_max_abs = std::max(local_max_abs, std::max(std::abs(p.x), std::max(std::abs(p.y), std::abs(p.z))));
            }

            #pragma omp critical
            {
                max_abs = std::max(max_abs, local_max_abs);
            }
        }

        // Calculate scaling factor to fit within [-1, 1]
        return 1.0f / max_abs; // Scaling factor to fit within [-1, 1]
    }

    std::array<std::pair<float, float>, 3> FastGoICP::get_point_cloud_ranges(PointCloud& pc)
    {
        // Initialize min and max values for x, y, z
        std::array<std::pair<float, float>, 3> ranges = {
            std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()),
            std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()),
            std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest())
        };

        #pragma omp parallel
        {
            // Local min and max for each thread
            std::array<std::pair<float, float>, 3> localRanges = {
                std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()),
                std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()),
                std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest())
            };

            #pragma omp for nowait
            for (size_t i = 0; i < pc.size(); ++i)
            {
                const auto& point = pc[i];
                localRanges[0].first = std::min(localRanges[0].first, point.x);
                localRanges[0].second = std::max(localRanges[0].second, point.x);

                localRanges[1].first = std::min(localRanges[1].first, point.y);
                localRanges[1].second = std::max(localRanges[1].second, point.y);

                localRanges[2].first = std::min(localRanges[2].first, point.z);
                localRanges[2].second = std::max(localRanges[2].second, point.z);
            }

            #pragma omp critical
            {
                ranges[0].first = std::min(ranges[0].first, localRanges[0].first);
                ranges[0].second = std::max(ranges[0].second, localRanges[0].second);

                ranges[1].first = std::min(ranges[1].first, localRanges[1].first);
                ranges[1].second = std::max(ranges[1].second, localRanges[1].second);

                ranges[2].first = std::min(ranges[2].first, localRanges[2].first);
                ranges[2].second = std::max(ranges[2].second, localRanges[2].second);
            }
        }

        return ranges;
    }


    float FastGoICP::scale_point_clouds(PointCloud& pct, PointCloud& pcs)
    {
        // Find the scaling factor for pcs
        float scaling_factor = get_scaling_factor(pcs);
        
        #pragma omp parallel for
        for (size_t i = 0; i < pcs.size(); ++i) 
        {
            pcs[i] *= scaling_factor;
        }
        #pragma omp parallel for
        for (size_t i = 0; i < pct.size(); ++i)
        {
            pct[i] *= scaling_factor;
        }
        return scaling_factor;
    }
}