#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <random>
#include "../core/tree.h"
#include "node.h"
#include "party.h"

struct RandomForestTree : Tree<RandomForestNode>
{
    RandomForestTree() {}
    void fit(vector<RandomForestParty> *parties, vector<float> *y,
             int num_classes, int min_leaf, int depth, vector<float> *prior,
             float max_samples_ratio = 1.0, float mi_bound = 0,
             int active_party_id = -1, int n_job = 1, int seed = 0)
    {
        vector<int> idxs(y->size());
        iota(idxs.begin(), idxs.end(), 0);

        if (max_samples_ratio < 1.0)
        {
            mt19937 engine(seed);
            shuffle(idxs.begin(), idxs.end(), engine);
            int temp_subsampled_size = int(max_samples_ratio * float(y->size()));
            idxs.resize(temp_subsampled_size);
        }

        for (int i = 0; i < parties->size(); i++)
        {
            parties->at(i).subsample_columns();
        }
        num_row = y->size();
        dtree = RandomForestNode(parties, y, num_classes, idxs, depth, prior, mi_bound, active_party_id, false, n_job);
    }

    void free_intermediate_resources()
    {
        dtree.y->clear();
        dtree.y->shrink_to_fit();
    }
};
