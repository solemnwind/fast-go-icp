#ifndef FGOICP_HPP
#define FGOICP_HPP

#include "common.hpp"

namespace icp
{
    class PointCloudRegistration
    {
        public:
            PointCloudRegistration(std::string config_path) : config{config_path} {};
            ~PointCloudRegistration() {};

            void initialize();
            //void run();

        private:
            icp::Config config;

    };
}

#endif // FGOICP_HPP