#ifndef FGOICP_HPP
#define FGOICP_HPP

#include "common.hpp"
#include <iostream>

namespace icp
{
    class PointCloudRegistration
    {
    public:
        PointCloudRegistration(std::string config_path) : config{config_path} {};
        ~PointCloudRegistration()
        {
            if (pcs) { delete[] pcs; }
            if (pct) { delete[] pct; }
        }

        void initialize();
        void run();

    private:
        icp::Config config;

        // Data
        size_t ns, nt; // number of source/target points
        Point3D *pcs;  // source point cloud
        Point3D *pct;  // target point cloud

        // Results
        float best_err;
        

        void build_kd_tree();

        void branch_and_bound();
    };
}

#endif // FGOICP_HPP