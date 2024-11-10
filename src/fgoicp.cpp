#include "fgoicp.hpp"
#include <iostream>

namespace icp
{
    void PointCloudRegistration::initialize()
    {
        Point3D* pcs;    // source point cloud
        Point3D* pct;    // target point cloud
        size_t ns = load_cloud_ply(config.io.source, pcs);
        size_t nt = load_cloud_ply(config.io.target, pct);
        std::cout << "Source points: " << ns << "\n"
                  << "Target points: " << nt << std::endl;
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

    return 0;
}
