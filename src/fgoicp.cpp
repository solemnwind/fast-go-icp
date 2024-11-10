#include "fgoicp.hpp"
#include <iostream>

namespace icp
{
    void PointCloudRegistration::initialize()
    {
        ns = load_cloud_ply(config.io.source, pcs, config.subsample);
        nt = load_cloud_ply(config.io.target, pct, 1.0);  // Never subsample target cloud
        std::cout << "Source points: " << ns << "\n"
                  << "Target points: " << nt << std::endl;
    } 

    void PointCloudRegistration::run()
    {
        branch_and_bound();
    }
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    icp::PointCloudRegistration pcr(argv[1]);
    pcr.initialize();
    pcr.run();

    return 0;
}
