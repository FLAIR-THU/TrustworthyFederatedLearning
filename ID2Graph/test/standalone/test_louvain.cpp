#include <limits>
#include <vector>
#include "llatvfl/louvain/louvain.h"
#include "gtest/gtest.h"
using namespace std;

TEST(Louvain, DenseTest)
{
    vector<vector<int>> test_adj_mat = {{0, 0, 1, 0, 0, 0, 0, 1},
                                        {0, 0, 0, 0, 1, 1, 0, 0},
                                        {1, 0, 0, 0, 0, 0, 0, 1},
                                        {0, 0, 0, 0, 0, 0, 1, 0},
                                        {0, 1, 0, 0, 0, 1, 0, 0},
                                        {0, 1, 0, 0, 1, 0, 0, 0},
                                        {0, 0, 0, 1, 0, 0, 0, 0},
                                        {1, 0, 1, 0, 0, 0, 0, 0}};

    int num_nodes = test_adj_mat.size();
    vector<unsigned long> degrees;
    vector<unsigned int> links;
    vector<float> weights;

    int cum_degree = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        for (int j = 0; j < num_nodes; j++)
        {
            if (test_adj_mat[i][j] != 0)
            {
                cum_degree += 1;
                links.push_back(j);
                weights.push_back(test_adj_mat[i][j]);
            }
        }
        degrees.push_back(cum_degree);
    }

    Graph g = Graph(num_nodes, degrees, links, weights);

    ASSERT_EQ(g.get_num_neighbors(0), 2);
    ASSERT_EQ(g.get_num_neighbors(3), 1);
    ASSERT_EQ(g.get_num_selfloops(0), 0);
    ASSERT_EQ(g.get_weighted_degree(0), 2);
    ASSERT_EQ(g.get_weighted_degree(2), 2);
    ASSERT_EQ(g.total_weight, 14);

    Community c = Community(&g, -1, 0.000001);
    ASSERT_NEAR(c.modularity(), -0.1326530612244898, 1e-6);
    c.compute_neigh_comms(0);
    ASSERT_EQ(c.neigh_last, 3);

    vector<float> test_neigh_weight = {0, -1, 1, -1, -1, -1, -1, 1};
    for (int i = 0; i < c.num_nodes; i++)
    {
        ASSERT_EQ(c.neigh_weight[i], test_neigh_weight[i]);
    }
}

TEST(Louvain, SparseTest)
{
    vector<vector<float>> test_adj_mat = {{0, 0, 1, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 1, 1, 0, 0},
                                          {1, 0, 0, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 0},
                                          {0, 1, 0, 0, 0, 1, 0, 0},
                                          {0, 1, 0, 0, 1, 0, 0, 0},
                                          {0, 0, 0, 1, 0, 0, 0, 0},
                                          {1, 0, 1, 0, 0, 0, 0, 0}};

    int num_nodes = test_adj_mat.size();
    SparseMatrixDOK<float> sm = SparseMatrixDOK<float>(num_nodes, num_nodes, 0, false, true);
    sm.from_densematrix(test_adj_mat);

    Graph g = Graph(sm);

    ASSERT_EQ(g.get_num_neighbors(0), 2);
    ASSERT_EQ(g.get_num_neighbors(3), 1);
    ASSERT_EQ(g.get_num_selfloops(0), 0);
    ASSERT_EQ(g.get_weighted_degree(0), 2);
    ASSERT_EQ(g.get_weighted_degree(2), 2);
    ASSERT_EQ(g.total_weight, 14);

    Community c = Community(&g, -1, 0.000001);
    ASSERT_NEAR(c.modularity(), -0.1326530612244898, 1e-6);
    c.compute_neigh_comms(0);
    ASSERT_EQ(c.neigh_last, 3);

    vector<float> test_neigh_weight = {0, -1, 1, -1, -1, -1, -1, 1};
    for (int i = 0; i < c.num_nodes; i++)
    {
        ASSERT_EQ(c.neigh_weight[i], test_neigh_weight[i]);
    }
}