#include "utilities.hpp"
#include <fgoicp/fgoicp.hpp>
#include <CLI11.hpp>
#include <glm/vec3.hpp>
#include <chrono>
#include <string>

int main(int argc, char* argv[])
{
    std::string config_file;
    bool verbose = false;

    CLI::App app_fgoicp{ "Fast Go-ICP: A CUDA Implementation of Go-ICP" };

    // Add options
    app_fgoicp.add_option("-c,--config", config_file, "Path to the TOML configuration file")
        ->required();

    app_fgoicp.add_flag("-v,--verbose", verbose, "Enable verbose logging");

    std::string exe_name = std::filesystem::path(argv[0]).filename().string();
    app_fgoicp.footer("\nExample Usage:\n"
        "  " + exe_name + " -c config.toml --verbose\n"
        "  " + exe_name + " --config=config.toml\n");

    // Parse the command-line arguments
    try {
        app_fgoicp.parse(argc, argv);
    }
    catch (const CLI::ParseError& e) {
        icp::Logger(icp::LogLevel::Error) << e.what();
        icp::Logger(icp::LogLevel::Info) << app_fgoicp.help();
        exit(e.get_exit_code());
    }

    icp::Logger::set_verbose(verbose);

    Config config(config_file);

    std::vector<glm::vec3> pct, pcs;
    load_cloud(config.io.target, config.params.target_subsample, pct);
    icp::Logger(icp::LogLevel::Info) << "Target point cloud (" << pct.size() << ") loaded from " << config.io.target;
    load_cloud(config.io.source, config.params.source_subsample, pcs);
    icp::Logger(icp::LogLevel::Info) << "Source point cloud (" << pcs.size() << ") loaded from " << config.io.source;

    icp::FastGoICP fgoicp(std::move(pct), std::move(pcs), 
                          config.params.lut_resolution, 
                          config.params.mse_threshold);

    auto start = std::chrono::high_resolution_clock::now();
    auto [R, t] = fgoicp.run();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    icp::Logger(icp::LogLevel::Info) << "Fast Go-ICP finished, time elapsed: " << std::fixed << std::setprecision(3)
        << elapsed_seconds.count() << " seconds";

    return 0;
}
