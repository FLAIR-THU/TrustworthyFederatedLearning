#include <algorithm>
#include <limits>
#include <vector>
#include <iostream>
#include "llatvfl/attack/attack.h"
#include "gtest/gtest.h"
using namespace std;

TEST(RandomForest, RandomForestClassifierTest)
{
    const int min_leaf = 1;
    const int depth = 2;
    const int num_trees = 1;
    const float subsample_cols = 1.0;
    const float max_samples_ratio = 1.0;

    // --- Load Data --- //
    int num_row = 8;
    int num_col = 2;
    int num_party = 2;

    vector<float> y = {1, 0, 1, 0, 1, 1, 0, 1};
    vector<vector<float>> X = {{12, 1},
                               {32, 1},
                               {15, 0},
                               {24, 0},
                               {20, 1},
                               {25, 1},
                               {17, 0},
                               {16, 1}};
    vector<vector<int>> feature_idxs = {{0}, {1}};
    vector<RandomForestParty> parties(num_party);

    for (int i = 0; i < num_party; i++)
    {
        int num_col = feature_idxs[i].size();
        vector<vector<float>> x(num_row, vector<float>(num_col));
        for (int j = 0; j < num_col; j++)
        {
            for (int k = 0; k < num_row; k++)
            {
                x[k][j] = X[k][feature_idxs[i][j]];
            }
        }
        RandomForestParty party(x, 2, feature_idxs[i], i, min_leaf, subsample_cols);
        parties[i] = party;
    }

    // --- Check Initialization --- //
    RandomForestClassifier clf = RandomForestClassifier(2, subsample_cols, depth, min_leaf,
                                                        max_samples_ratio, num_trees,
                                                        numeric_limits<float>::infinity(), -1, 1);

    // --- Check Training --- //
    clf.fit(parties, y);

    ASSERT_NEAR(clf.estimators[0].dtree.giniimp, 0.46875, 1e-6);
    ASSERT_NEAR(clf.estimators[0].dtree.score, 0.16875, 1e-6);
    ASSERT_EQ(clf.estimators[0].dtree.best_party_id, 0);
    ASSERT_EQ(clf.estimators[0].dtree.best_col_id, 0);
    ASSERT_EQ(clf.estimators[0].dtree.best_threshold_id, 2);

    ASSERT_EQ(clf.estimators[0].dtree.party_id, 0);
    ASSERT_EQ(get<0>(clf.estimators[0].dtree.parties->at(
                                                        clf.estimators[0].dtree.party_id)
                         .lookup_table.at(clf.estimators[0].dtree.record_id)),
              0);
    ASSERT_EQ(get<1>(clf.estimators[0].dtree.parties->at(
                                                        clf.estimators[0].dtree.party_id)
                         .lookup_table.at(clf.estimators[0].dtree.record_id)),
              16);

    vector<int> test_idxs_left = {0, 2, 7};
    // vector<int> test_idxs_right = {1, 3, 4, 5, 6};
    vector<int> idxs_left = clf.estimators[0].dtree.left->idxs;
    sort(idxs_left.begin(), idxs_left.end());
    ASSERT_EQ(idxs_left.size(), test_idxs_left.size());
    for (int i = 0; i < idxs_left.size(); i++)
    {
        ASSERT_EQ(idxs_left[i], test_idxs_left[i]);
    }

    /*
    vector<int> idxs_right = clf.estimators[0].dtree.right->idxs;
    sort(idxs_right.begin(), idxs_right.end());
    ASSERT_EQ(idxs_right.size(), test_idxs_right.size());
    for (int i = 0; i < idxs_right.size(); i++)
    {
        ASSERT_EQ(idxs_right[i], test_idxs_right[i]);
    }
    */

    ASSERT_EQ(clf.estimators[0].dtree.right->depth, 1);
    ASSERT_EQ(clf.estimators[0].dtree.left->is_leaf(), 1);
    ASSERT_EQ(clf.estimators[0].dtree.right->is_leaf(), 0);

    ASSERT_EQ(clf.estimators[0].dtree.right->party_id, 1);
    ASSERT_EQ(get<1>(clf.estimators[0].dtree.right->parties->at(
                                                               clf.estimators[0].dtree.right->party_id)
                         .lookup_table.at(clf.estimators[0].dtree.right->record_id)),
              0);

    vector<int> test_idxs_right_left = {3, 6};
    vector<int> test_idxs_right_right = {1, 4, 5};
    vector<int> idxs_right_left = clf.estimators[0].dtree.right->left->idxs;
    vector<int> idxs_right_right = clf.estimators[0].dtree.right->right->idxs;
    sort(idxs_right_left.begin(), idxs_right_left.end());
    sort(idxs_right_right.begin(), idxs_right_right.end());
    for (int i = 0; i < test_idxs_right_left.size(); i++)
    {
        ASSERT_EQ(test_idxs_right_left[i], idxs_right_left[i]);
    }
    for (int i = 0; i < test_idxs_right_right.size(); i++)
    {
        ASSERT_EQ(test_idxs_right_right[i], idxs_right_right[i]);
    }

    vector<float> test_predict_raw = {1, 2.0 / 3.0, 1, 0, 2.0 / 3.0, 2.0 / 3.0, 0, 1};
    vector<vector<float>> predict_raw = clf.predict_raw(X);
    for (int i = 0; i < predict_raw.size(); i++)
    {
        ASSERT_EQ(test_predict_raw[i], predict_raw[i][1]);
    }

    vector<float> test_predict_proba = {1, 2.0 / 3.0, 1, 0, 2.0 / 3.0, 2.0 / 3.0, 0, 1};
    vector<vector<float>> predict_proba = clf.predict_proba(X);
    for (int i = 0; i < predict_proba.size(); i++)
    {
        ASSERT_NEAR(test_predict_proba[i], predict_proba[i][1], 1e-6);
    }

    vector<vector<float>> test_adj_mat = {{0, 0, 1, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 1, 1, 0, 0},
                                          {1, 0, 0, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 0},
                                          {0, 1, 0, 0, 0, 1, 0, 0},
                                          {0, 1, 0, 0, 1, 0, 0, 0},
                                          {0, 0, 0, 1, 0, 0, 0, 0},
                                          {1, 0, 1, 0, 0, 0, 0, 0}};

    clf.free_intermediate_resources();
    SparseMatrixDOK<float> adj_mat_sparse = extract_adjacency_matrix_from_forest(&clf, false, -1, false);
    vector<vector<float>> adj_mat = adj_mat_sparse.to_densematrix();
    for (int j = 0; j < test_adj_mat.size(); j++)
    {
        for (int k = 0; k < test_adj_mat[j].size(); k++)
        {
            ASSERT_EQ(adj_mat[j][k], test_adj_mat[j][k]);
        }
    }

    // ASSERT_NEAR(adj_mat_sparse.get_nonzero_ratio(), 0.21875, 1e-6);
}