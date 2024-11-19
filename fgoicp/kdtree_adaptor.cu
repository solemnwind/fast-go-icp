#include "kdtree_adaptor.hpp"
#include <cmath>

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
}
