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
#include <cuda.h>
#include <cuda_runtime.h>

namespace icp
{
    /**
     * @brief DataSource definition for `nanoflann` adaptor
     * 
     */
    struct _DataSource 
    {
        const PointCloud &points;

        _DataSource(const PointCloud &points) : points(points) {}

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

    /**
     * @brief `nanoflann` K-D Tree Adaptor for 3D `float` with L2-norm metric
     * 
     */
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, _DataSource>,
        _DataSource,
        3 /* Data dimensionality */
    > KDTree;

    /**
     * @brief ArrayNode for flattened K-D Tree
     * 
     */
    struct ArrayNode 
    {
        bool is_leaf;

        union {
            // Leaf node data
            struct {
                size_t left, right;  // Indices of points in the leaf node
            } leaf;

            // Non-leaf node data
            struct {
                int divfeat;           // Dimension used for subdivision
                float divlow, divhigh; // Range values used for subdivision
                size_t child1, child2; // Indices of child nodes in the array
            } nonleaf;
        } data;
    };

    void flatten_kd_tree(KDTree kdt, float* arr_kdt);

    __device__ void find_nearest_neighbor(float* arr_kdt);
}

#endif // KDTREE_ADAPTOR_HPP
