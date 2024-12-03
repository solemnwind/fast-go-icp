#include "utilities.hpp"
#include "fgoicp/fgoicp.hpp"
#include <glm/vec3.hpp>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    Config config(argv[1]);
    
    std::vector<glm::vec3> pct, pcs;
    load_cloud(config.io.target, 1.0, pct);
    icp::Logger(icp::LogLevel::Info) << "Target point cloud loaded from " << config.io.target;
    load_cloud(config.io.source, config.subsample, pcs);
    icp::Logger(icp::LogLevel::Info) 
        << "Source point cloud loaded from " << config.io.source
        << "\tSubsample rate: " << config.subsample;

    icp::FastGoICP fgoicp(std::move(pct), std::move(pcs), config.mse_threshold);
    fgoicp.run();

    return 0;
}
