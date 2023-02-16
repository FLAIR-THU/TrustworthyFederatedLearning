#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "../core/model.h"
#include "tree.h"
using namespace std;

struct RandomForestClassifier : TreeModelBase<RandomForestParty>
{
    float subsample_cols;
    int num_classes;
    int depth;
    int min_leaf;
    float max_samples_ratio;
    int num_trees;
    float mi_bound;
    int active_party_id;
    int n_job;
    int seed;

    vector<RandomForestTree> estimators;

    RandomForestClassifier(int num_classes_ = 2, float subsample_cols_ = 0.8, int depth_ = 5, int min_leaf_ = 1,
                           float max_samples_ratio_ = 1.0, int num_trees_ = 5,
                           float mi_bound_ = numeric_limits<float>::infinity(),
                           int active_party_id_ = -1, int n_job_ = 1, int seed_ = 0)
    {
        num_classes = num_classes_;
        subsample_cols = subsample_cols_;
        depth = depth_;
        min_leaf = min_leaf_;
        max_samples_ratio = max_samples_ratio_;
        num_trees = num_trees_;
        mi_bound = mi_bound_;
        active_party_id = active_party_id_;
        n_job = n_job_;
        seed = seed_;

        if (mi_bound < 0)
        {
            mi_bound = numeric_limits<float>::infinity();
        }
    }

    void load_estimators(vector<RandomForestTree> &_estimators)
    {
        estimators = _estimators;
    }

    void clear()
    {
        estimators.clear();
    }

    vector<RandomForestTree> get_estimators()
    {
        return estimators;
    }

    void fit(vector<RandomForestParty> &parties, vector<float> &y)
    {
        int row_count = y.size();

        vector<float> prior(num_classes, 0);
        for (int j = 0; j < row_count; j++)
        {
            prior[y[j]] += 1;
        }
        for (int c = 0; c < num_classes; c++)
        {
            prior[c] /= float(row_count);
        }

        for (int i = 0; i < num_trees; i++)
        {
            RandomForestTree tree = RandomForestTree();
            tree.fit(&parties, &y, num_classes, min_leaf, depth, &prior, max_samples_ratio, mi_bound, active_party_id, n_job, seed);
            estimators.push_back(tree);
            seed += 1;
        }
    }

    // retuen the average score of all trees (sklearn-style)
    vector<vector<float>> predict_raw(vector<vector<float>> &X)
    {
        int row_count = X.size();
        vector<vector<float>> y_pred(row_count, vector<float>(num_classes, 0));
        int estimators_num = estimators.size();
        for (int i = 0; i < estimators_num; i++)
        {
            vector<vector<float>> y_pred_temp = estimators[i].predict(X);
            for (int j = 0; j < row_count; j++)
            {
                for (int c = 0; c < num_classes; c++)
                {
                    y_pred[j][c] += y_pred_temp[j][c] / float(estimators_num);
                }
            }
        }

        return y_pred;
    }

    vector<vector<float>> predict_proba(vector<vector<float>> &x)
    {
        return predict_raw(x);
    }

    void free_intermediate_resources()
    {
        int estimators_num = estimators.size();
        for (int i = 0; i < estimators_num; i++)
        {
            estimators[i].free_intermediate_resources();
        }
    }
};
