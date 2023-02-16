#include <limits>
#include <vector>
#include "llatvfl/attack/attack.h"
#include "gtest/gtest.h"
#include "iostream"
using namespace std;

TEST(XGBoost, XGBoostClassifierTest)
{
    const int min_leaf = 1;
    const int depth = 3;
    const float learning_rate = 0.4;
    const int boosting_rounds = 2;
    const float lam = 1.0;
    const float const_gamma = 0.0;
    const float eps = 1.0;
    const float min_child_weight = -1 * numeric_limits<float>::infinity();
    const float subsample_cols = 1.0;

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
    vector<XGBoostParty> parties(num_party);

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
        XGBoostParty party(x, 2, feature_idxs[i], i, min_leaf, subsample_cols);
        parties[i] = party;
    }

    // --- Check Initialization --- //
    XGBoostClassifier clf = XGBoostClassifier(2, subsample_cols,
                                              min_child_weight,
                                              depth, min_leaf,
                                              learning_rate,
                                              boosting_rounds,
                                              lam, const_gamma, eps,
                                              numeric_limits<float>::infinity(),
                                              0, 0, 1.0, 1);

    vector<float> test_init_pred = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    vector<vector<float>> init_pred = clf.get_init_pred(y);
    for (int i = 0; i < init_pred.size(); i++)
        ASSERT_EQ(init_pred[i][1], test_init_pred[i]);

    vector<vector<float>> base_pred;
    copy(init_pred.begin(), init_pred.end(), back_inserter(base_pred));
    vector<float> test_base_grad = {-0.26894, 0.73106, -0.26894, 0.73106,
                                    -0.26894, -0.26894, 0.73106, -0.26894};
    vector<vector<float>> grad = clf.lossfunc_obj->get_grad(base_pred, y);
    for (int i = 0; i < grad.size(); i++)
        ASSERT_NEAR(grad[i][0], test_base_grad[i], 1e-5);

    vector<vector<float>> hess = clf.lossfunc_obj->get_hess(base_pred, y);
    vector<float> test_hess = {0.19661, 0.19661, 0.19661, 0.19661,
                               0.19661, 0.19661, 0.19661, 0.19661};
    for (int i = 0; i < hess.size(); i++)
        ASSERT_NEAR(hess[i][0], test_hess[i], 1e-5);

    // --- Check Training --- //
    clf.fit(parties, y);
    clf.free_intermediate_resources();

    ASSERT_EQ(parties[0].get_lookup_table().size(), 4);
    ASSERT_EQ(parties[1].get_lookup_table().size(), 2);

    ASSERT_EQ(clf.estimators[0].dtree.get_num_parties(), 2);

    // vector<int> test_idxs_root = {0, 1, 2, 3, 4, 5, 6, 7};
    // vector<int> idxs_root = clf.estimators[0].dtree.idxs;
    // for (int i = 0; i < idxs_root.size(); i++)
    //     ASSERT_EQ(idxs_root[i], test_idxs_root[i]);
    ASSERT_EQ(clf.estimators[0].dtree.idxs.size(), 0);
    ASSERT_EQ(clf.estimators[0].dtree.idxs.capacity(), 0);

    ASSERT_EQ(clf.estimators[0].dtree.gradient->size(), 0);
    ASSERT_EQ(clf.estimators[0].dtree.gradient->capacity(), 0);
    ASSERT_EQ(clf.estimators[0].dtree.left->gradient->size(), 0);
    ASSERT_EQ(clf.estimators[0].dtree.left->gradient->capacity(), 0);

    ASSERT_EQ(clf.estimators[0].dtree.depth, 3);
    ASSERT_EQ(get<0>(clf.estimators[0].dtree.parties->at(clf.estimators[0].dtree.party_id).lookup_table.at(clf.estimators[0].dtree.record_id)), 0);
    ASSERT_EQ(get<1>(clf.estimators[0].dtree.parties->at(clf.estimators[0].dtree.party_id).lookup_table.at(clf.estimators[0].dtree.record_id)), 16);
    ASSERT_EQ(clf.estimators[0].dtree.is_leaf(), 0);

    vector<int> test_idxs_left = {0, 2, 7};
    vector<int> idxs_left = clf.estimators[0].dtree.left->idxs;
    for (int i = 0; i < idxs_left.size(); i++)
        ASSERT_EQ(idxs_left[i], test_idxs_left[i]);
    ASSERT_TRUE(clf.estimators[0].dtree.left->is_pure());
    ASSERT_TRUE(clf.estimators[0].dtree.left->is_leaf());
    ASSERT_NEAR(clf.estimators[0].dtree.left->val[0], 0.5074890528001861, 1e-6);

    // vector<int> test_idxs_right = {1, 3, 4, 5, 6};
    // vector<int> idxs_right = clf.estimators[0].dtree.right->idxs;
    // for (int i = 0; i < idxs_right.size(); i++)
    //    ASSERT_EQ(idxs_right[i], test_idxs_right[i]);
    ASSERT_TRUE(!clf.estimators[0].dtree.right->is_pure());
    ASSERT_TRUE(!clf.estimators[0].dtree.right->is_leaf());
    ASSERT_NEAR(clf.estimators[0].dtree.right->val[0], -0.8347166357912786, 1e-6);
    XGBoostNode right_node = *clf.estimators[0].dtree.right;
    ASSERT_EQ(right_node.party_id, 1);
    ASSERT_EQ(get<0>(right_node.parties->at(right_node.party_id)
                         .lookup_table.at(right_node.record_id)),
              0);

    XGBoostNode right_right_node = *right_node.right;
    ASSERT_EQ(right_right_node.party_id, 0);
    ASSERT_EQ(get<0>(right_right_node.parties->at(right_right_node.party_id)
                         .lookup_table.at(right_right_node.record_id)),
              0);
    ASSERT_EQ(get<1>(right_right_node.parties->at(right_right_node.party_id)
                         .lookup_table.at(right_right_node.record_id)),
              25);

    ASSERT_TRUE(clf.estimators[0].dtree.right->right->left->is_leaf());
    ASSERT_TRUE(clf.estimators[0].dtree.right->right->right->is_leaf());
    ASSERT_NEAR(clf.estimators[0].dtree.right->right->left->val[0], 0.3860706492904221, 1e-6);
    ASSERT_NEAR(clf.estimators[0].dtree.right->right->right->val[0], -0.6109404045885225, 1e-6);

    vector<vector<float>> predict_raw = clf.predict_raw(X);
    vector<float> test_predcit_raw = {1.38379341, 0.53207456, 1.38379341,
                                      0.22896408, 1.29495549, 1.29495549,
                                      0.22896408, 1.38379341};
    for (int i = 0; i < test_predcit_raw.size(); i++)
        ASSERT_NEAR(predict_raw[i][0], test_predcit_raw[i], 1e-6);

    vector<vector<float>> predict_proba = clf.predict_proba(X);
    vector<float> test_predcit_proba = {0.79959955, 0.62996684, 0.79959955,
                                        0.55699226, 0.78498478, 0.78498478,
                                        0.55699226, 0.79959955};
    for (int i = 0; i < test_predcit_proba.size(); i++)
    {
        ASSERT_NEAR(predict_proba[i][1], test_predcit_proba[i], 1e-6);
        ASSERT_NEAR(predict_proba[i][0], 1 - test_predcit_proba[i], 1e-6);
    }

    /*
    vector<vector<float>> test_adj_mat = {{0, 0, 1.3, 0, 0, 0, 0, 1.3},
                                          {0, 0, 0, 0, 0, 0, 0, 0},
                                          {1.3, 0, 0, 0, 0, 0, 0, 1.3},
                                          {0, 0, 0, 0, 0, 0, 1.3, 0},
                                          {0, 0, 0, 0, 0, 1.3, 0, 0},
                                          {0, 0, 0, 0, 1.3, 0, 0, 0},
                                          {0, 0, 0, 1.3, 0, 0, 0, 0},
                                          {1.3, 0, 1.3, 0, 0, 0, 0, 0}};

    clf.free_intermediate_resources();
    vector<vector<float>> adj_mat = extract_adjacency_matrix_from_forest(&clf, depth, -1, 0, 0.3).to_densematrix();
    for (int j = 0; j < test_adj_mat.size(); j++)
    {
        for (int k = 0; k < test_adj_mat[j].size(); k++)
        {
            ASSERT_EQ(adj_mat[j][k], test_adj_mat[j][k]);
        }
    }
    */

    vector<vector<float>> test_adj_mat_1 = {{0, 0, 1.3, 0, 0, 0, 0, 1.3},
                                            {0, 0, 0, 0, 1.3, 1.3, 0, 0},
                                            {1.3, 0, 0, 0, 0, 0, 0, 1.3},
                                            {0, 0, 0, 0, 0, 0, 1.3, 0},
                                            {0, 1.3, 0, 0, 0, 1.3, 0, 0},
                                            {0, 1.3, 0, 0, 1.3, 0, 0, 0},
                                            {0, 0, 0, 1.3, 0, 0, 0, 0},
                                            {1.3, 0, 1.3, 0, 0, 0, 0, 0}};

    vector<vector<float>> adj_mat_1 = extract_adjacency_matrix_from_forest(&clf, false, 1, 0, 0.3).to_densematrix();
    for (int j = 0; j < test_adj_mat_1.size(); j++)
    {
        for (int k = 0; k < test_adj_mat_1[j].size(); k++)
        {
            ASSERT_EQ(adj_mat_1[j][k], test_adj_mat_1[j][k]);
        }
    }
}