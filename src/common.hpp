#ifndef COMMON_HPP
#define COMMON_HPP
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

using std::string;

namespace icp
{
    const float PI = 3.141592653589793f;
    const float INF = 1e10f;

    struct Rotation
    {
        float rr, x, y, z;

        // Precompute coefficients for rotation computation
        float r11, r12, r13;
        float r21, r22, r23;
        float r31, r32, r33;

        Rotation(float x, float y, float z) : 
            x(x), y(y), z(z)
        {
            rr = x * x + y * y + z * z;
            if (rr > 1.0f) { return; } // Not a rotation

            float ww = 1.0f - rr;
            float w = sqrt(ww);
            float wx = w * x; float xx = x * x;
            float wy = w * y; float xy = x * y; float yy = y * y;
            float wz = w * z; float xz = x * z; float yz = y * z; float zz = z * z;
            r11 = (ww + xx - yy - zz);  r12 = 2 * (xy - wz);        r13 = 2 * (xz + wy); 
            r21 = 2 * (xy + wz);        r22 = (ww - xx + yy - zz);  r23 = 2 * (yz - wx);
            r31 = 2 * (xz - wy);        r32 = 2 * (yz + wx);        r33 = (ww - xx - yy + zz);
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
            return q.w >= 0.0f;
        }
    };

    struct Vector
    {
        float x, y, z;
    };

    /**
     * @brief Translation Node in the R(3) space.
     * 
     */
    struct TransNode
    {
        Vector t;
        float span;
        float ub, lb;
    };

    typedef Vector Point3D;

    /**
     * @brief Rotate and translate a point
     * 
     * @param p Point3D p
     * @param q Rotation q (Rotation)
     * @param t Vector t (Translation)
     */
    Point3D transform_SE3(const Point3D &p, const Rotation &q, const Vector &t)
    {
        float a = q.r11 * p.x + q.r12 * p.y + q.r13 * p.z + t.x;
        float b = q.r21 * p.x + q.r22 * p.y + q.r23 * p.z + t.y;
        float c = q.r31 * p.x + q.r32 * p.y + q.r33 * p.z + t.z;
        return Point3D{a, b, c};
    }
    
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
