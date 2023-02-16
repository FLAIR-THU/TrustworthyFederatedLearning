#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "../core/tree.h"
#include "node.h"

/**
 * @brief Tree structure of XGBoost
 *
 */
struct XGBoostTree : Tree<XGBoostNode>
{
    XGBoostTree() {}

    /**
     * @brief Train a single XGBoost tree.
     *
     * @param parties
     * @param y
     * @param num_classes
     * @param gradient
     * @param hessian
     * @param prior
     * @param min_child_weight
     * @param lam
     * @param gamma
     * @param eps
     * @param min_leaf
     * @param depth
     * @param mi_bound
     * @param active_party_id
     * @param use_only_active_party
     * @param n_job
     */
    void fit(vector<XGBoostParty> *parties, vector<float> *y, int num_classes,
             vector<vector<float>> *gradient, vector<vector<float>> *hessian,
             vector<float> *prior, float min_child_weight, float lam,
             float gamma, float eps, int min_leaf, int depth, float mi_bound,
             int active_party_id = -1, bool use_only_active_party = false, int n_job = 1)
    {
        vector<int> idxs(y->size());
        iota(idxs.begin(), idxs.end(), 0);
        for (int i = 0; i < parties->size(); i++)
        {
            parties->at(i).subsample_columns();
        }
        num_row = y->size();
        dtree = XGBoostNode(parties, y, num_classes, gradient, hessian, idxs, prior,
                            min_child_weight, lam, gamma, eps, depth, mi_bound,
                            active_party_id, use_only_active_party, n_job);
    }

    void free_intermediate_resources()
    {
        dtree.y->clear();
        dtree.y->shrink_to_fit();
    }
};
