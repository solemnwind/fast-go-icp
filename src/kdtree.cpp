#include "kdtree.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace icp
{
    KDTree::KDTree(const Point3D* points, size_t num_points)
    {
        std::cout << "Building K-D Tree..." << std::endl;
        using std::chrono::high_resolution_clock;
        using std::chrono::duration_cast;
        using std::chrono::duration;
        using std::chrono::milliseconds;

        auto t1 = high_resolution_clock::now();
        root = build(points, 0, num_points, 0);
        auto t2 = high_resolution_clock::now();

        /* Getting number of milliseconds as an integer. */
        auto ms_int = duration_cast<milliseconds>(t2 - t1);

        /* Getting number of milliseconds as a double. */
        duration<double, std::milli> ms_double = t2 - t1;

        std::cout << ms_int.count() << "ms\n";
        std::cout << ms_double.count() << "ms\n";
        dev_root = nullptr;     // Initially no device memory allocated
    }

    KDTree::Node* KDTree::build(const Point3D* points, size_t start, size_t end, int depth)
    {
        if (start >= end) return nullptr;

        int axis = depth % 3; // x=0, y=1, z=2

        // Sort points based on the selected axis
        switch (axis) {
            case 0: std::nth_element(points + start, points + (start + end) / 2, points + end, [](const Point3D& a, const Point3D& b) { return a.x < b.x; }); break;
            case 1: std::nth_element(points + start, points + (start + end) / 2, points + end, [](const Point3D& a, const Point3D& b) { return a.y < b.y; }); break;
            case 2: std::nth_element(points + start, points + (start + end) / 2, points + end, [](const Point3D& a, const Point3D& b) { return a.z < b.z; }); break;
        }

        size_t median = (start + end) / 2;
        Node* node = new Node(points[median]);

        node->left = build(points, start, median, depth + 1);
        node->right = build(points, median + 1, end, depth + 1);

        return node;
    }

}