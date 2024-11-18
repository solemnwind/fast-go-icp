#ifndef FGOICP_HPP
#define FGOICP_HPP

#include "common.hpp"
#include "registration.hpp"
#include <iostream>

namespace icp
{
    class PointCloudRegistration
    {
    public:
        PointCloudRegistration(std::string config_path) : 
            config{config_path}, pcs(), pct(),
            ns{load_cloud_ply(config.io.source, config.subsample, pcs)},
            nt{load_cloud_ply(config.io.target, 1.0, pct)},
            reg{pct, nt, pcs, ns}
        {
            std::cout << "Source points: " << ns << "\t"
                      << "Target points: " << nt << std::endl;
        };

        ~PointCloudRegistration() {}

        void run();

    private:
        icp::Config config;

        // Data
        PointCloud pcs;  // source point cloud
        PointCloud pct;  // target point cloud
        size_t ns, nt; // number of source/target points

        Registration reg;

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