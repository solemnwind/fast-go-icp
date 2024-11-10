#include "common.hpp"
#include <iostream>
#include "toml.hpp"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

namespace icp
{
    Config::Config(const string toml_filepath) :
        io{"", "", "", ""},
        rotation{0, 0, 0, 0, 0, 0},
        translation{0, 0, 0, 0, 0, 0}
    { 
        string base_filename = toml_filepath.substr(toml_filepath.find_last_of("/\\") + 1);
        std::cout << "Reading configurations from " << base_filename << std::endl;
        int err = parse_toml(toml_filepath);
        if (err) { throw; }
    }

    int Config::parse_toml(const string toml_filepath)
    {
        toml::table tbl;
        try
        {
            tbl = toml::parse_file(toml_filepath);
        }
        catch (const toml::parse_error& err)
        {
            std::cerr
                << "Error parsing file '" << *err.source().path
                << "':\n" << err.description()
                << "\n (" << err.source().begin << ")\n";
            return 1;
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

        return 0;
    }

    size_t load_cloud_ply(const string ply_filepath, Point3D* cloud)
    {
        size_t num_points = 0;

        try
        {
            std::ifstream file_stream(ply_filepath, std::ios::binary);
            if (!file_stream)
            {
                throw std::runtime_error("Unable to open file: " + ply_filepath);
            }

            // Load the PLY file
            tinyply::PlyFile ply_file;
            ply_file.parse_header(file_stream);

            // Prepare buffers for x, y, z data
            std::shared_ptr<tinyply::PlyData> vertices;

            try
            {
                vertices = ply_file.request_properties_from_element("vertex", { "x", "y", "z" });
            }
            catch (const std::exception &err)
            {
                throw std::runtime_error("PLY file missing 'x', 'y', or 'z' vertex properties.");
            }

            // Read the file data
            ply_file.read(file_stream);

            // Check that we have valid data
            if (vertices && vertices->count > 0)
            {
                num_points = vertices->count;

                // Dynamically allocate memory for points
                cloud = new Point3D[num_points];

                const float *vertex_buffer = reinterpret_cast<const float *>(vertices->buffer.get());

                for (size_t i = 0; i < num_points; ++i)
                {
                    cloud[i].x = vertex_buffer[3 * i + 0];
                    cloud[i].y = vertex_buffer[3 * i + 1];
                    cloud[i].z = vertex_buffer[3 * i + 2];
                }
            }
            else
            {
                throw std::runtime_error("No vertices found in the PLY file.");
            }
        }
        catch (const std::exception &err)
        {
            std::cerr << "Error reading PLY file: " << err.what() << std::endl;
        }

        return num_points;
    }
}
