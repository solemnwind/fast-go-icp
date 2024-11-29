#ifndef ICP3D_HPP
#define ICP3D_HPP

#include "common.hpp"
#include "registration.hpp"

namespace icp
{
    class IterativeClosestPoint3D
    {
    private:
        const Registration& reg;
        const size_t nt;
        const size_t ns;
        glm::mat3 R;
        glm::vec3 t;
        const size_t max_iter;
        const float sse_threshold;
        const float convergence_threshold;

        PointCloud _pct_buffer;
        PointCloud _pcs_buffer;
        Point3D* d_pct_buffer;
        Point3D* d_pcs_buffer;
        Point3D* d_corrs_buffer;
        Point3D* d_pcs_centered_buffer;
        Point3D* d_corrs_centered_buffer;
        glm::mat3* d_mat_buffer;

    public:
        IterativeClosestPoint3D(const Registration& reg, const PointCloud& pct, const PointCloud& pcs, size_t max_iter, float sse_threshold, glm::mat3 R, glm::vec3 t);

        ~IterativeClosestPoint3D();

        using Result_t = std::tuple<float, glm::mat3, glm::vec3>;
        Result_t run();

    private:
        using RigidMotion_t = std::tuple<glm::mat3, glm::vec3>;
        RigidMotion_t procrustes();
    };
}

#endif // ICP3D_HPP
