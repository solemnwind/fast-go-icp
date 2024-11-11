/**
 * @file kdtree_adaptor.hpp
 * @author Zhaojin Sun () & Mufeng Xu (mufeng.xu@outlook.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef KDTREE_ADAPTOR_HPP
#define KDTREE_ADAPTOR_HPP

#include "nanoflann.hpp"
#include "common.hpp"
#include <vector>
#include <iostream>

namespace icp
{
    // Tree struct definition for nanoflann usage
    struct Tree 
    {
        PointCloud points;

        // Required methods for nanoflann
        inline size_t kdtree_get_point_count() const 
        {
            return points.size();
        }

        inline float kdtree_get_pt(const size_t idx, const size_t dim) const 
        {
            if (dim == 0) return points[idx].x;
            else if (dim == 1) return points[idx].y;
            else return points[idx].z;
        }

        // Distance computation for nanoflann
        template <class T>
        float kdtree_distance(const T* p1, const size_t idx_p2, size_t /*size*/) const 
        {
            const Point3D& p2 = points[idx_p2];
            float d0 = p1[0] - p2.x;
            float d1 = p1[1] - p2.y;
            float d2 = p1[2] - p2.z;
            return d0 * d0 + d1 * d1 + d2 * d2;  // Return squared distance
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /*bb*/) const 
        {
            return false; // Return false indicating no bounding box computation
        }
    };

    // Define the KDTree type
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, Tree>,
        Tree,
        3 /* Data dimensionality */
    > KDTree;

    // Function to build the KDTree and run an example query
    void buildKDTree(Tree& tree) 
    {
        // Ensure this function only runs on the host side
        std::unique_ptr<KDTree> kdtree = std::make_unique<KDTree>(
            3 /* Data dimensionality */, tree, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
        );
        kdtree->buildIndex();

        std::cout << "KDTree built with " << tree.kdtree_get_point_count() << " points." << std::endl;

        // Example: nearest neighbor search
        float query_pt[3] = { 1.0, 1.0, 1.0 }; // Example query point
        size_t num_results = 1;
        std::vector<size_t> ret_index(num_results);
        std::vector<float> out_dist_sqr(num_results);

        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        kdtree->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));

        std::cout << "Nearest neighbor index: " << ret_index[0] << ", squared distance: " << out_dist_sqr[0] << std::endl;
    }
}

#endif // KDTREE_ADAPTOR_HPP
