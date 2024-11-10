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
                if (pcs) delete[] pcs;
                if (pct) delete[] pct;
            }

            void initialize();
            //void run();

        private:
            icp::Config config;

            size_t ns, nt;   // number of source/target points
            Point3D* pcs;    // source point cloud
            Point3D* pct;    // target point cloud
    };
}

#endif // FGOICP_HPP