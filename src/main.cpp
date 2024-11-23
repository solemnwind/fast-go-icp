#include "fgoicp/fgoicp.hpp"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    icp::FastGoICP fgoicp(argv[1]);
    fgoicp.run();

    return 0;
}
