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
        ~PointCloudRegistration() {}

        void initialize();
        void run();

    private:
        icp::Config config;

        // Data
        size_t ns, nt; // number of source/target points
        PointCloud pcs;  // source point cloud
        PointCloud pct;  // target point cloud

        // Results
        float best_err;

        /**
         * @brief Perform branch-and-bound algorithm in SE(3) space
         * 
         * @details This is a __host__ function, 
         * it launches cuda kernels for translation BnB.
         * 
         * @return none
         */
        void branch_and_bound();
    };
}

#endif // FGOICP_HPP