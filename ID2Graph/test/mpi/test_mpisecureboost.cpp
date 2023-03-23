#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <numeric>
#include <string>
#include <cassert>
#include <future>
#include <utility>
#include <chrono>
#include <unistd.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include "llatvfl/secureboost/mpisecureboost.h"
#include "llatvfl/paillier/keygenerator.h"
#include "llatvfl/utils/metric.h"
#include "gtest/gtest.h"
using namespace std;

TEST(MPISecureBoost, MPISecureBoostClassifierTest)
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
    const int key_bitsize = 512;
    const int active_party_id = 0;

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

    boost::mpi::environment env(true);
    boost::mpi::communicator world;
    int my_rank = world.rank();
    MPISecureBoostParty my_party;

    for (int i = 0; i < num_party; i++)
    {
        if (i == my_rank)
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
            my_party = MPISecureBoostParty(world, x, 2, feature_idxs[i], my_rank, depth,
                                           boosting_rounds, min_leaf, subsample_cols,
                                           const_gamma, lam);
        }
    }

    MPISecureBoostClassifier clf = MPISecureBoostClassifier(2, subsample_cols,
                                                            min_child_weight,
                                                            depth, min_leaf,
                                                            learning_rate, boosting_rounds,
                                                            lam, const_gamma, eps,
                                                            0, 0, 1.0);

    if (my_rank == 0)
    {
        PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
        pair<PaillierPublicKey, PaillierSecretKey> keypair = keygenerator.generate_keypair();
        PaillierPublicKey pk = keypair.first;
        PaillierSecretKey sk = keypair.second;

        my_party.set_publickey(pk);
        my_party.set_secretkey(sk);

        for (int j = 1; j < num_party; j++)
        {
            world.send(j, TAG_PUBLICKEY, pk);
        }

        my_party.y = y;
    }
    else
    {
        PaillierPublicKey pk;
        world.recv(0, TAG_PUBLICKEY, pk);
        my_party.set_publickey(pk);
        my_party.pk.init_distribution();
    }

    world.barrier();

    clf.fit(my_party, num_party);

    vector<vector<float>> predict_raw = clf.predict_raw(X);
    vector<vector<float>> predict_proba = clf.predict_proba(X);

    if (my_rank == 0)
    {
        vector<float> test_init_pred = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        vector<vector<float>> init_pred = clf.get_init_pred(y);
        for (int i = 0; i < init_pred.size(); i++)
        {
            ASSERT_EQ(init_pred[i][0], test_init_pred[i]);
        }

        ASSERT_EQ(my_party.get_lookup_table().size(), 4);

        // vector<int> test_idxs_root = {0, 1, 2, 3, 4, 5, 6, 7};
        // vector<int> idxs_root = clf.estimators[0].dtree.idxs;
        // for (int i = 0; i < idxs_root.size(); i++)
        //{
        //     ASSERT_EQ(idxs_root[i], test_idxs_root[i]);
        // }

        ASSERT_EQ(clf.estimators[0].dtree.depth, 3);
        ASSERT_FALSE(clf.estimators[0].dtree.is_leaf());

        vector<int> test_idxs_left = {0, 2, 7};
        vector<int> idxs_left = clf.estimators[0].dtree.left->idxs;
        for (int i = 0; i < idxs_left.size(); i++)
            ASSERT_EQ(idxs_left[i], test_idxs_left[i]);
        ASSERT_TRUE(clf.estimators[0].dtree.left->is_pure());
        ASSERT_TRUE(clf.estimators[0].dtree.left->is_leaf());
        ASSERT_NEAR(clf.estimators[0].dtree.left->val[0], 0.5074890528001861, 1e-6);

        ASSERT_TRUE(clf.estimators[0].dtree.right->right->left->is_leaf());
        ASSERT_TRUE(clf.estimators[0].dtree.right->right->right->is_leaf());
        ASSERT_NEAR(clf.estimators[0].dtree.right->right->left->val[0], 0.3860706492904221, 1e-6);
        ASSERT_NEAR(clf.estimators[0].dtree.right->right->right->val[0], -0.6109404045885225, 1e-6);

        vector<float> test_predcit_raw = {1.38379341, 0.53207456, 1.38379341,
                                          0.22896408, 1.29495549, 1.29495549,
                                          0.22896408, 1.38379341};
        for (int i = 0; i < test_predcit_raw.size(); i++)
        {
            ASSERT_NEAR(predict_raw[i][0], test_predcit_raw[i], 1e-6);
        }

        vector<float> test_predcit_proba = {0.79959955, 0.62996684, 0.79959955,
                                            0.55699226, 0.78498478, 0.78498478,
                                            0.55699226, 0.79959955};
        for (int i = 0; i < test_predcit_raw.size(); i++)
        {
            ASSERT_NEAR(predict_proba[i][1], test_predcit_proba[i], 1e-6);
        }
    }
    else
    {
        ASSERT_EQ(my_party.get_lookup_table().size(), 2);
    }
}