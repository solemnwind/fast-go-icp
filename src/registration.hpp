#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"
#include "kdtree.hpp"

namespace icp
{
    class Registration
    {
    public:
        Registration(const Point3D* pct, size_t nt, const Point3D* pcs, size_t ns) : 
            kdt_target{pct, nt},
            pct(pct), pcs(pcs),
            nt(nt), ns(ns)
        {}

        ~Registration()
        {
            // free kdt_target;
        }

        /**
         * @brief Run single-step ICP registration algorithm.
         * 
         * @return float: MSE error
         */
        __host__ __device__ float run(Rotation &q, Vector &t);

    private:
        KDTree kdt_target;
        const Point3D* pct;
        const Point3D* pcs;
        const size_t nt;
        const size_t ns;

        size_t max_iter = 10;

    };

}

#endif // REGISTRATION_HPP