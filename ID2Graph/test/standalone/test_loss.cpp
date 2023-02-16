#include <limits>
#include <vector>
#include "gtest/gtest.h"
#include "llatvfl/xgboost/loss.h"
using namespace std;

TEST(Utils, CELoss)
{
    vector<vector<float>> input = {{1, 2, 3, 4, 5}};
    vector<float> output = {2};
    CELoss loss = CELoss(5);

    ASSERT_NEAR(loss.get_loss(input, output), 2.4519143105, 1e-4);

    vector<vector<float>> grad = loss.get_grad(input, output);
    vector<vector<float>> test_grad = {{0.0116562322, 0.0316849239, -0.9138714671, 0.2341216505,
                                        0.6364086270}};
    for (int i = 0; i < grad.size(); i++)
    {
        ASSERT_NEAR(grad[i][0], test_grad[i][0], 1e-4);
    }

    vector<vector<float>> hess = loss.get_hess(input, output);
    vector<vector<float>> test_hess = {{0.0115203634, 0.0306809861, 0.0787104145, 0.1793086976, 0.2313926816}};
    for (int i = 0; i < hess.size(); i++)
    {
        ASSERT_NEAR(hess[i][0], test_hess[i][0], 1e-4);
    }
}