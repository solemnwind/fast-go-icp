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
            pct(pct), pcs(pcs),
            nt(nt), ns(ns),
            kdt_target{3, _DataSource(pct), nanoflann::KDTreeSingleIndexAdaptorParams(10)}
        {
            // Create and build the KDTree
            kdt_target.buildIndex();

            std::cout << "KD-tree built with " << pct.size() << " points." << std::endl;
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
        /**
         * @brief Target point cloud
         */
        const PointCloud &pct;
        /**
         * @brief Source point cloud
         */
        const PointCloud &pcs;
        /**
         * @brief Number of target cloud points
         */
        const size_t nt;
        /**
         * @brief Number of source cloud points
         */
        const size_t ns;
        KDTree kdt_target;

        size_t max_iter = 10;

    };

}

#endif // REGISTRATION_HPP