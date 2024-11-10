#ifndef COMMON_HPP
#define COMMON_HPP
#include <string>

using std::string;

namespace icp 
{
    class Config
    {
        public:
            bool trim;
            float subsample;
            float mse_threshold;

            struct IO 
            {
                std::string target;         // target point cloud ply path
                std::string source;         // source point cloud ply path
                std::string output;         // output toml path
                std::string visualization;  // visualization ply path
            } io;

            struct Rotation
            {
                float xmin, xmax;
                float ymin, ymax;
                float zmin, zmax;
            } rotation;

            struct Translation
            {
                float xmin, xmax;
                float ymin, ymax;
                float zmin, zmax;
            } translation; 

            Config(const string toml_filepath);

        private:
            int parse_toml(const string toml_filepath);
    };

    struct Point3D
    {
        float x;
        float y;
        float z;
    };

    size_t load_cloud_ply(const string ply_filepath, Point3D* &cloud, const float subsample);
}

#endif // COMMON_HPP
