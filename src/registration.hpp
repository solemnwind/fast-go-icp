#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"
#include "kdtree_adaptor.hpp"

namespace icp
{
    class Registration
    {
    public:
        Registration(const PointCloud &pct, size_t nt, const PointCloud &pcs, size_t ns) : 
            kdt_target{3, tree, nanoflann::KDTreeSingleIndexAdaptorParams(10)},
            pct(pct), pcs(pcs),
            nt(nt), ns(ns)
        {
            // Load points into tree.points
            // Example: tree.points.push_back(glm::vec3(x, y, z));
            tree.points = pct;
            // Create and build the KDTree
            kdt_target.buildIndex();

            std::cout << "KD-tree built with " << tree.kdtree_get_point_count() << " points." << std::endl;
        }

        ~Registration()
        {
            // free kdt_target;
        }

        /**
         * @brief Run single-step ICP registration algorithm.
         * 
         * @return float: MSE error
         */
        __host__ __device__ float run(Rotation &q, glm::vec3 &t);

    private:
        Tree tree;
        KDTree kdt_target;
        const PointCloud &pct;
        const PointCloud &pcs;
        const size_t nt;
        const size_t ns;

        size_t max_iter = 10;

    };

}

#endif // REGISTRATION_HPP