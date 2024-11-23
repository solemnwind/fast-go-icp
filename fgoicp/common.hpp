#ifndef COMMON_HPP
#define COMMON_HPP
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#define M_PI    3.141592653589793f
#define M_INF   1E+10f
#define M_SQRT3 1.732050807568877f

using std::string;

namespace icp
{
    struct Rotation
    {
        float rr, x, y, z;
        glm::mat3 R;

        Rotation() : Rotation(0.0f, 0.0f, 0.0f) {}

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

        Rotation(Rotation &other) :
            rr(other.rr), x(other.x), y(other.y), z(other.z), R(other.R)
        {}
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

    size_t load_cloud_ply(const std::string& ply_filepath, const float& subsample, std::vector<glm::vec3>& cloud);
    size_t load_cloud_txt(const std::string& txt_filepath, const float& subsample, std::vector<glm::vec3>& cloud);
    size_t load_cloud(const std::string& filepath, const float& subsample, std::vector<glm::vec3>& cloud);

    enum class LogLevel 
    {
        Debug,
        Info,
        Warning,
        Error
    };

    class Logger 
    {
    public:
        explicit Logger(LogLevel level) : level_(level) {}

        // Overload << operator for streaming
        template <typename T>
        Logger& operator<<(const T& msg) 
        {
            buffer_ << msg; // Stream the message into the buffer
            return *this;
        }

        // Destructor: automatically flushes and prints the log when the object goes out of scope
        ~Logger() 
        {
            std::string color, prefix;
            switch (level_) 
            {
                case LogLevel::Debug:
                    color = "\033[34m"; // Blue
                    prefix = "[Debug " + get_current_time() + "] ";
                    break;
                case LogLevel::Info:
                    color = "\033[32m"; // Green
                    prefix = "[Info " + get_current_time() + "] ";
                    break;
                case LogLevel::Warning:
                    color = "\033[33m"; // Yellow
                    prefix = "[Warning " + get_current_time() + "] ";
                    break;
                case LogLevel::Error:
                    color = "\033[31m"; // Red
                    prefix = "[Error " + get_current_time() + "] ";
                    break;
            }
            // Print the final message with color
            std::cout << color << prefix << buffer_.str() << "\033[0m" << "\n";
        }

    private:
        LogLevel level_;
        std::ostringstream buffer_;

        // Helper function to get the current timestamp
        std::string get_current_time() 
        {
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::tm buf{}; 
#if defined(_WIN32) || defined(_WIN64)
            localtime_s(&buf, &in_time_t); // Thread-safe on Windows
#else
            localtime_r(&in_time_t, &buf); // Thread-safe on Linux
#endif
            std::ostringstream ss;
            ss << std::put_time(&buf, "%H:%M:%S");
            return ss.str();
        }
    };
}

#endif // COMMON_HPP
