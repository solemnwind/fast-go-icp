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

#define CUDA_DEBUG 0

#define M_PI    3.141592653589793f
#define M_INF   1E+10f
#define M_SQRT3 1.732050807568877f

using std::string;

namespace icp
{
    void cudaCheckError(string info, bool silent = true);

    //========================================================================================
    //                           Rotation, Translation & PointCloud
    //========================================================================================
    struct Rotation
    {
        float x, y, z, r;
        glm::mat3 R;

        Rotation() : Rotation(0.0f, 0.0f, 0.0f) {}

        Rotation(float x, float y, float z) : 
            x(x), y(y), z(z), 
            r(x * x + y * y + z * z),
            R(1.0f)
        {
            if (r > 1.0f) { return; } // Not a rotation

            float ww = 1.0f - r;
            float w = sqrt(ww);
            float wx = w * x, xx = x * x;
            float wy = w * y, xy = x * y, yy = y * y;
            float wz = w * z, xz = x * z, yz = y * z, zz = z * z;

            R = glm::mat3(
                ww + xx - yy - zz,  2 * (xy - wz),      2 * (xz + wy),
                2 * (xy + wz),      ww - xx + yy - zz,  2 * (yz - wx),
                2 * (xz - wy),      2 * (yz + wx),      ww - xx - yy + zz
            );

            r = sqrt(r);
        }

        Rotation(const Rotation &other) :
            r(other.r), x(other.x), y(other.y), z(other.z), R(other.R)
        {}

        /**
         * @brief Validate that the Rotation is in SO(3)
         * 
         * @return true if valid; false if not
         */
        bool in_SO3() const { return r <= 1.0f; }
    };

    /**
     * @brief Rotation Node in the bounding box where SO(3) resides in.
     * 
     */
    struct RotNode
    {
        Rotation q;     // Coordinate in the bounding box
        float span;     // half edge length of the bounding cube of a rotation node
        float lb, ub;   // upper and lower error bound of this node

        RotNode(float x, float y, float z, float span, float lb, float ub) :
            q(x, y, z), span(span), lb(lb), ub(ub)
        {}

        friend bool operator<(const RotNode& rnode1, const RotNode& rnode2)
        {
            if (rnode1.lb == rnode2.lb)
            {
                return rnode1.span < rnode2.span;
            }
            return rnode1.lb > rnode2.lb;
        }

        /**
         * @brief Validate that the Rotation Node is (partially) in SO(3)
         * 
         * @return true if valid; false if not
         */
        bool overlaps_SO3() const
        {
            // (|x|-s)^2 + (|y|-s)^2 + (|y|-s)^2 <= 1
            return q.r - 2 * span * (abs(q.x) + abs(q.y) + abs(q.z)) + 3 * span * span <= 1;
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
        float lb, ub;

        TransNode(float x, float y, float z, float span, float lb, float ub) :
            t(x, y, z), span(span), lb(lb), ub(ub)
        {}

        friend bool operator<(const TransNode& tnode1, const TransNode& tnode2)
        {
            if (tnode1.lb == tnode2.lb)
            {
                return tnode1.span < tnode2.span;
            }
            return tnode1.lb > tnode2.lb;
        }
    };

    typedef glm::vec3 Point3D;

    using PointCloud = std::vector<Point3D>;
    
    //========================================================================================
    //                                    CUDA Stream Pool
    //========================================================================================

    class StreamPool 
    {
    public:
        explicit StreamPool(size_t size) : streams(size) 
        {
            for (auto& stream : streams) 
            {
                cudaStreamCreate(&stream);
            }
        }

        ~StreamPool() 
        {
            for (auto& stream : streams) 
            {
                cudaStreamDestroy(stream);
            }
        }

        cudaStream_t getStream(size_t index) const 
        {
            return streams[index % streams.size()];
        }

    private:
        std::vector<cudaStream_t> streams;
    };


    //========================================================================================
    //                                     Fancy Logger
    //========================================================================================

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
        Logger() : Logger(LogLevel::Debug) {}

        // Overload << operator for streaming
        template <typename T>
        Logger& operator<<(const T& msg) 
        {
            buffer_ << msg; // Stream the message into the buffer
            return *this;
        }

        // Overload << for glm::vec3
        Logger& operator<<(const glm::vec3& vec)
        {
            buffer_ << std::fixed << std::setprecision(6);
            buffer_ << vec.x << "\t" << vec.y << "\t" << vec.z;
            return *this;
        }

        // Overload << for glm::mat3
        Logger& operator<<(const glm::mat3& mat)
        {
            buffer_ << std::fixed << std::setprecision(4);
            buffer_ << "\t" << mat[0][0] << "\t" << mat[1][0] << "\t" << mat[2][0] << "\n";
            buffer_ << "\t" << mat[0][1] << "\t" << mat[1][1] << "\t" << mat[2][1] << "\n";
            buffer_ << "\t" << mat[0][2] << "\t" << mat[1][2] << "\t" << mat[2][2];
            return *this;
        }

        // Destructor: automatically flushes and prints the log when the object goes out of scope
        ~Logger() 
        {
            if (level_ == LogLevel::Debug && !verbose_) 
            {
                return; // Skip Debug logs if quiet mode is enabled
            }

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

        static void set_verbose(bool verbose)
        {
            verbose_ = verbose;
        }

    private:
        LogLevel level_;
        std::ostringstream buffer_;

        static bool verbose_;

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
