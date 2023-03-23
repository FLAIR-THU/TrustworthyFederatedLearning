#include <cmath>
#include <vector>
#include "llatvfl/utils/metric.h"
#include "gtest/gtest.h"
using namespace std;

TEST(Metric, AUCTest)
{
    vector<int> y_true = {0, 0, 0, 0, 1, 1, 1, 1};
    vector<float> y_pred = {0.2, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 0.9};
    ASSERT_EQ(roc_auc_score(y_pred, y_true), 0.6875);
}