#include <vector>
#include "llatvfl/utils/dok.h"
#include "gtest/gtest.h"

TEST(DOK, HashTest)
{
    HashPairSzudzik hashfunc;
    ASSERT_EQ(hashfunc(make_pair(92, 23)), 8579);
}

TEST(DOK, DOKTest)
{
    SparseMatrixDOK<float> sm = SparseMatrixDOK<float>(3, 3, 0, false, true);
    sm.add(1, 1, 0.5);
    sm.add(2, 1, 0.1);
    sm.add(1, 1, -0.5);
    sm.add(2, 1, 0.3);
    sm.add(0, 2, 1);

    vector<vector<float>> test_adj_mat = {{0, 0, 1.0},
                                          {0, 0, 0},
                                          {0, 0.4, 0}};
    vector<vector<float>> adj_mat = sm.to_densematrix(0);

    for (int i = 0; i < test_adj_mat.size(); i++)
    {
        assert(adj_mat.size() == test_adj_mat.size());
        for (int j = 0; j < test_adj_mat[i].size(); j++)
        {
            ASSERT_EQ(sm(i, j), test_adj_mat[i][j]);
            ASSERT_EQ(adj_mat[i][j], test_adj_mat[i][j]);
        }
    }

    vector<vector<int>> test_row2nonzero_idx = {{2}, {}, {1}};
    for (int i = 0; i < test_row2nonzero_idx.size(); i++)
    {
        for (int j = 0; j < test_row2nonzero_idx[i].size(); j++)
        {
            ASSERT_EQ(test_row2nonzero_idx[i][j], sm.row2nonzero_idx[i][j]);
        }
    }

    sm = SparseMatrixDOK<float>(3, 3, 0, false, true);
    sm.from_densematrix(test_adj_mat);
    vector<vector<float>> reconstructed_dm = sm.to_densematrix();
    for (int i = 0; i < test_adj_mat.size(); i++)
    {
        for (int j = 0; j < test_adj_mat[i].size(); j++)
        {
            ASSERT_EQ(test_adj_mat[i][j], reconstructed_dm[i][j]);
        }
    }

    SparseMatrixDOK<float> sm_symme = SparseMatrixDOK<float>(3, 3, 0, true, true);
    sm_symme.add(1, 1, 0.5);
    sm_symme.add(2, 1, 0.1);
    sm_symme.add(1, 1, -0.5);
    sm_symme.add(2, 1, 0.3);
    sm_symme.add(0, 2, 1);

    vector<vector<float>> test_adj_mat_symme = {{0, 0, 1.0},
                                                {0, 0, 0.4},
                                                {1.0, 0.4, 0}};
    vector<vector<float>> adj_mat_symme = sm_symme.to_densematrix(0);

    for (int i = 0; i < test_adj_mat_symme.size(); i++)
    {
        assert(adj_mat_symme.size() == test_adj_mat_symme.size());
        for (int j = 0; j < test_adj_mat_symme[i].size(); j++)
        {
            ASSERT_EQ(sm_symme(i, j), test_adj_mat_symme[i][j]);
            ASSERT_EQ(adj_mat_symme[i][j], test_adj_mat_symme[i][j]);
        }
    }

    vector<vector<int>> test_row2nonzero_idx_symme = {{}, {}, {1, 0}};
    for (int i = 0; i < test_row2nonzero_idx_symme.size(); i++)
    {
        for (int j = 0; j < test_row2nonzero_idx_symme[i].size(); j++)
        {
            ASSERT_EQ(test_row2nonzero_idx_symme[i][j], sm_symme.row2nonzero_idx[i][j]);
        }
    }

    sm_symme = SparseMatrixDOK<float>(3, 3, 0, true, true);
    sm_symme.from_densematrix(test_adj_mat_symme);
    vector<vector<float>> reconstructed_dm_symme = sm_symme.to_densematrix();
    for (int i = 0; i < test_adj_mat_symme.size(); i++)
    {
        for (int j = 0; j < test_adj_mat_symme[i].size(); j++)
        {
            ASSERT_EQ(test_adj_mat_symme[i][j], reconstructed_dm_symme[i][j]);
        }
    }
}