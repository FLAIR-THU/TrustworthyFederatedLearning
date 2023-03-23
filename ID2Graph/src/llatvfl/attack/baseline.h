#pragma once
#include "../randomforest/randomforest.h"
#include "../secureboost/secureboost.h"
#include "../utils/dok.h"
#include "../xgboost/xgboost.h"
#include <iostream>
#include <limits>
#include <queue>
#include <vector>
using namespace std;

struct UnionTree {
  vector<int> parents, size;
  UnionTree(int n) {
    parents.resize(n, 0);
    size.resize(n, 0);
    for (int i = 0; i < n; i++) {
      makeTree(i);
    }
  }

  void makeTree(int x) {
    parents[x] = x;
    size[x] = 1;
  }

  int findRoot(int x) {
    if (parents[x] != x) {
      parents[x] = findRoot(parents[x]);
    }
    return parents[x];
  }

  bool isSame(int x, int y) { return findRoot(x) == findRoot(y); }

  bool unite(int x, int y) {
    x = findRoot(x);
    y = findRoot(y);

    if (x == y) {
      return false;
    } else {
      if (size[x] > size[y]) {
        parents[y] = x;
        size[x] += size[y];
      } else {
        parents[x] = y;
        size[y] += size[x];
      }
      return true;
    }
  }
};

template <typename NodeType>
inline void travase_nodes_to_extract_uniontree(NodeType *node,
                                               UnionTree &uinontree,
                                               int target_party_id) {
  queue<NodeType *> que;
  que.push(node);
  NodeType *temp_node;
  int temp_idxs_size;
  while (!que.empty()) {
    // skip_flag = false;
    temp_node = que.front();
    que.pop();

    if (temp_node->is_leaf_flag) {
      if (!temp_node->not_splitted_flag || target_party_id == -1) {
        temp_idxs_size = temp_node->idxs.size();
        for (int i = 1; i < temp_idxs_size; i++) {
          uinontree.unite(temp_node->idxs[0], temp_node->idxs[i]);
        }
      }
    } else {
      bool not_splitted_flag = temp_node->left->not_splitted_flag &&
                               temp_node->right->not_splitted_flag;
      bool lmir_exclude_flag =
          temp_node->left->lmir_flag_exclude_passive_parties &&
          temp_node->right->lmir_flag_exclude_passive_parties;
      bool exclude_flag =
          (not_splitted_flag || lmir_exclude_flag) && (target_party_id != -1);

      if (exclude_flag) {
        temp_idxs_size = temp_node->idxs.size();
        for (int i = 1; i < temp_idxs_size; i++) {
          uinontree.unite(temp_node->idxs[0], temp_node->idxs[i]);
        }
      }

      if (!temp_node->left->lmir_flag_exclude_passive_parties ||
          !temp_node->right->lmir_flag_exclude_passive_parties) {
        que.push(temp_node->left);
        que.push(temp_node->right);
      }
    }

    temp_node->idxs.clear();
    temp_node->idxs.shrink_to_fit();
    temp_node->val.clear();
    temp_node->val.shrink_to_fit();
  }
}

inline void extract_uniontree_from_tree(
    XGBoostTree *tree, UnionTree &uinontree, int target_party_id) {
  travase_nodes_to_extract_uniontree<XGBoostNode>(&tree->dtree, uinontree,
                                                  target_party_id);
}

inline void extract_uniontree_from_tree(
    RandomForestTree *tree, UnionTree &uinontree, int target_party_id) {
  travase_nodes_to_extract_uniontree<RandomForestNode>(&tree->dtree, uinontree,
                                                  target_party_id);
}

inline void extract_uniontree_from_tree(
    SecureBoostTree *tree, UnionTree &uinontree, int target_party_id) {
  travase_nodes_to_extract_uniontree<SecureBoostNode>(&tree->dtree, uinontree,
                                                  target_party_id);
}

template <typename ModelType>
inline vector<int> extract_uniontree_from_forest(ModelType *model,
                                                 int target_party_id = -1,
                                                 int skip_round = 0)
{
    int num_row = model->estimators[0].num_row;
    UnionTree uniontree(num_row);
    for (int i = 0; i < model->estimators.size(); i++)
    {
        if (i >= skip_round)
        {
            extract_uniontree_from_tree(&model->estimators[i], uniontree, target_party_id);
        }
    }

    vector<int> result(num_row, 0);
    for (int j = 0; j < num_row; j++){
      result[j] = uniontree.findRoot(j);
    }

    return result;
}