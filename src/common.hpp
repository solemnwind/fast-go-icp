#ifndef COMMON_HPP
#define COMMON_HPP
#include <string>

using std::string;

namespace icp
{
    /**
     * @brief Rotation Node in the bounding box where SO(3) resides in.
     * 
     */
    struct RotNode
    {
        float x, y, z;  // Coordinate in the bounding box
        float span;     // Span of the node
        float ub, lb;   // upper and lower error bound of this node 
    };

    /**
     * @brief Translation Node in the R(3) space.
     * 
     */
    struct TransNode
    {
        float x, y, z;
        float span;
        float ub, lb;
    };

    struct Point3D
    {
        float x;
        float y;
        float z;
    };
    
    class Config
    {
    public:
        bool trim;
        float subsample;
        float mse_threshold;

        struct IO
        {
            std::string target;        // target point cloud ply path
            std::string source;        // source point cloud ply path
            std::string output;        // output toml path
            std::string visualization; // visualization ply path
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
        void parse_toml(const string toml_filepath);
    };

    size_t load_cloud_ply(const string ply_filepath, Point3D *&cloud, const float subsample);
}

#endif // COMMON_HPP
