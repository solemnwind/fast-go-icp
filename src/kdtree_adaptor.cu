#include "kdtree_adaptor.hpp"

namespace icp
{
    void FlattenedKDTree::flatten_KDTree(const KDTree::Node* root, std::vector<ArrayNode>& array, size_t& currentIndex) 
    {
        if (root == nullptr) return;

        size_t index = currentIndex++;
        array.resize(index + 1);
        
        if (root->child1 == nullptr && root->child2 == nullptr) {
            // Leaf node
            array[index].is_leaf = true;
            array[index].data.leaf.left = root->node_type.lr.left;
            array[index].data.leaf.right = root->node_type.lr.right;
        } else {
            // Non-leaf node
            array[index].is_leaf = false;
            array[index].data.nonleaf.divfeat = root->node_type.sub.divfeat;
            array[index].data.nonleaf.divlow = root->node_type.sub.divlow;
            array[index].data.nonleaf.divhigh = root->node_type.sub.divhigh;

            // Recursively flatten left and right child nodes
            size_t child1Index = currentIndex;
            flatten_KDTree(root->child1, array, currentIndex);
            array[index].data.nonleaf.child1 = child1Index;

            size_t child2Index = currentIndex;
            flatten_KDTree(root->child2, array, currentIndex);
            array[index].data.nonleaf.child2 = child2Index;
        }
    }

    void FlattenedKDTree::convert_KDTree_to_array(const KDTree &kdt) 
    {
        size_t currentIndex = 0;
        flatten_KDTree(kdt.root_node_, array, currentIndex);
    }

    #include <cmath>

    float distanceSquared(const Point3D &point1, const Point3D &point2) {
        float dist = 0;
        for (size_t i = 0; i < 3; i++) {
            float diff = point1[i] - point2[i];
            dist += diff * diff;
        }
        return dist;
    }

    void FlattenedKDTree::find_nearest_neighbor(const Point3D &target, size_t index, 
                    float& bestDist, size_t& bestIndex, int depth = 0) {
        if (index >= array.size()) return;

        const ArrayNode& node = array[index];
        if (node.is_leaf) {
            // Leaf node: Check all points in the leaf node
            size_t left = node.data.leaf.left;
            size_t right = node.data.leaf.right;
            for (size_t i = left; i <= right; i++) {
                float dist = distanceSquared(target, pct[vAcc_[i]]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIndex = vAcc_[i];
                }
            }
        } else {
            // Non-leaf node: Determine which child to search
            int axis = node.data.nonleaf.divfeat;
            float diff = target[axis] - node.data.nonleaf.divlow;

            // Choose the near and far child based on comparison
            size_t nearChild = diff < 0 ? node.data.nonleaf.child1 : node.data.nonleaf.child2;
            size_t farChild = diff < 0 ? node.data.nonleaf.child2 : node.data.nonleaf.child1;

            // Search near child
            find_nearest_neighbor(target, nearChild, bestDist, bestIndex, depth + 1);

            // Search far child if needed
            if (diff * diff < bestDist) {
                find_nearest_neighbor(target, farChild, bestDist, bestIndex, depth + 1);
            }
        }
    }


}