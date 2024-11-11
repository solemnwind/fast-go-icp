#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"

namespace icp
{
    /**
     * @brief Implementation of KDTree<float>
     * 
     */
    class KDTree
    {
    public:
        struct Node
        {
            Point3D point;
            Node* left;
            Node* right;

            Node() : point(), left(nullptr), right(nullptr) {}
            Node(Point3D pt) : point(pt), left(nullptr), right(nullptr) {}
        };

        KDTree(const Point3D* points, size_t num_points);
        // ~KDTree();

        /**
         * @brief Queries the nearest neighbor in the KD-Tree
         * 
         * @param p Point3D* : pointer to the query point p
         * @return float: distance to the nearest point
         */
        __device__ __host__ float query(const Point3D* const point);
        
    private:
        Node* root;
        Node* dev_root;

        Node* build(const Point3D* points, size_t start, size_t end, int depth);

        void free_tree();
    };
}

#endif // KDTREE_HPP