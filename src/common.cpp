#include "common.hpp"
#include <iostream>
#include <random>
#include "toml.hpp"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#define clamp(x, min, max) (x) < (max) ? ((x) > (min) ? (x) : (min)) : (max)

namespace icp
{
    Config::Config(const string toml_filepath)
        : trim(false), subsample(1.0f), mse_threshold(1e-5f),
          io{"", "", "", ""},
          rotation{0, 0, 0, 0, 0, 0},
          translation{0, 0, 0, 0, 0, 0}
    {
        string base_filename = toml_filepath.substr(toml_filepath.find_last_of("/\\") + 1);
        std::cout << "Reading configurations from " << base_filename << std::endl;
        parse_toml(toml_filepath);
    }

    void Config::parse_toml(const string toml_filepath)
    {
        toml::table tbl;

        try
        {
            tbl = toml::parse_file(toml_filepath);
        }
        catch (const toml::parse_error &err)
        {
            std::string err_msg = "Error parsing file '";
            err_msg += *err.source().path;  // Dereference the pointer to get the path as a string
            err_msg += "': ";
            err_msg += err.description();
            err_msg += "\n";

            throw std::runtime_error(err_msg);
        }
 

        std::optional<string> desc = tbl["info"]["description"].value<string>();
        std::cout << desc.value() << std::endl;

        // Parse IO section
        if (tbl.contains("io"))
        {
            auto io_section = tbl["io"];
            io.target = io_section["target"].value_or("");
            io.source = io_section["source"].value_or("");
            io.output = io_section["output"].value_or("");
            io.visualization = io_section["visualization"].value_or("");
        }

        // Parse parameters
        if (tbl.contains("params"))
        {
            auto params_section = tbl["params"];
            trim = params_section["trim"].value_or(false);
            subsample = params_section["subsample"].value_or(1.0f);
            mse_threshold = params_section["mse_threshold"].value_or(1e-5f);

            // Check bounding conditions
            subsample = clamp(subsample, 0.0f, 1.0f);
            mse_threshold = clamp(mse_threshold, 1e-10f, INFINITY);
        }

        // Parse Rotation section
        if (tbl.contains("params") && tbl["params"].as_table()->contains("rotation"))
        {
            auto rotation_section = tbl["params"]["rotation"];
            rotation.xmin = rotation_section["xmin"].value_or(-1.0f);
            rotation.xmax = rotation_section["xmax"].value_or(1.0f);
            rotation.ymin = rotation_section["ymin"].value_or(-1.0f);
            rotation.ymax = rotation_section["ymax"].value_or(1.0f);
            rotation.zmin = rotation_section["zmin"].value_or(-1.0f);
            rotation.zmax = rotation_section["zmax"].value_or(1.0f);
        }

        // Parse Translation section
        if (tbl.contains("params") && tbl["params"].as_table()->contains("translation"))
        {
            auto translation_section = tbl["params"]["translation"];
            translation.xmin = translation_section["xmin"].value_or(-1.0f);
            translation.xmax = translation_section["xmax"].value_or(1.0f);
            translation.ymin = translation_section["ymin"].value_or(-1.0f);
            translation.ymax = translation_section["ymax"].value_or(1.0f);
            translation.zmin = translation_section["zmin"].value_or(-1.0f);
            translation.zmax = translation_section["zmax"].value_or(1.0f);
        }

        std::cout << "Config parsed successfully." << std::endl;
    }

    size_t load_cloud_ply(const string ply_filepath, Point3D *&cloud, const float subsample)
    {
        size_t num_points = 0;

        try
        {
            std::ifstream file_stream(ply_filepath, std::ios::binary);
            if (!file_stream)
            {
                throw std::runtime_error("Unable to open file: " + ply_filepath);
            }

            tinyply::PlyFile ply_file;
            ply_file.parse_header(file_stream);

            std::shared_ptr<tinyply::PlyData> vertices;

            try
            {
                vertices = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
            }
            catch (const std::exception &err)
            {
                throw std::runtime_error("PLY file missing 'x', 'y', or 'z' vertex properties.");
            }

            ply_file.read(file_stream);

            if (vertices && vertices->count > 0)
            {
                size_t total_points = vertices->count;
                num_points = static_cast<size_t>(total_points * subsample);

                cloud = new Point3D[num_points];
                const float *vertex_buffer = reinterpret_cast<const float *>(vertices->buffer.get());

                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> dis(0.0, 1.0);

                size_t index = 0;
                for (size_t i = 0; i < total_points && index < num_points; ++i)
                {
                    if (dis(gen) <= subsample)
                    {
                        cloud[index].x = vertex_buffer[3 * i + 0];
                        cloud[index].y = vertex_buffer[3 * i + 1];
                        cloud[index].z = vertex_buffer[3 * i + 2];
                        ++index;
                    }
                }

                // Adjust num_points if fewer points were randomly selected
                num_points = index;
            }
            else
            {
                throw std::runtime_error("No vertices found in the PLY file.");
            }
        }
        catch (const std::exception &err)
        {
            throw std::runtime_error(string("Error reading PLY file: ") + err.what());
        }

        return num_points;
    }
}
