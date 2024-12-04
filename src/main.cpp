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
    
    icp::Logger(icp::LogLevel::Info) << "Subsample rate: " << config.subsample;

    std::vector<glm::vec3> pct, pcs;
    load_cloud(config.io.target, config.subsample, pct);
    icp::Logger(icp::LogLevel::Info) << "Target point cloud loaded from " << config.io.target;
    load_cloud(config.io.source, config.subsample, pcs);
    icp::Logger(icp::LogLevel::Info) << "Source point cloud loaded from " << config.io.source;

    icp::FastGoICP fgoicp(std::move(pct), std::move(pcs), config.mse_threshold);
    auto [R, t] = fgoicp.run();

    return 0;
}
