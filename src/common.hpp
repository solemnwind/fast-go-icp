#ifndef COMMON_HPP
#define COMMON_HPP
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

using std::string;

namespace icp
{
    const float PI = 3.141592653589793f;
    const float INF = 1e10f;

    struct Rotation
    {
        float rr, x, y, z;
        glm::mat3 R;

        Rotation(float x, float y, float z) : 
            x(x), y(y), z(z)
        {
            rr = x * x + y * y + z * z;
            if (rr > 1.0f) { return; } // Not a rotation

            float ww = 1.0f - rr;
            float w = sqrt(ww);
            float wx = w * x, xx = x * x;
            float wy = w * y, xy = x * y, yy = y * y;
            float wz = w * z, xz = x * z, yz = y * z, zz = z * z;

            R = glm::mat3(
                ww + xx - yy - zz,  2 * (xy - wz),      2 * (xz + wy),
                2 * (xy + wz),      ww - xx + yy - zz,  2 * (yz - wx),
                2 * (xz - wy),      2 * (yz + wx),      ww - xx - yy + zz
            );
        }
    };

    /**
     * @brief Rotation Node in the bounding box where SO(3) resides in.
     * 
     */
    struct RotNode
    {
        Rotation q;  // Coordinate in the bounding box
        float span;     // Span of the node
        float ub, lb;   // upper and lower error bound of this node

        bool is_valid()
        {
            return q.rr <= 1.0f;
        }
    };

    /**
     * @brief Translation Node in the R(3) space.
     * 
     */
    struct TransNode
    {
        glm::vec3 t;
        float span;
        float ub, lb;
    };

    typedef glm::vec3 Point3D;

    using PointCloud = std::vector<Point3D>;
    
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

    size_t load_cloud_ply(const string ply_filepath, const float subsample, PointCloud &cloud);
}

#endif // COMMON_HPP
