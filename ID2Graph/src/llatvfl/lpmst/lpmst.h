#pragma once
#include <algorithm>
#include "rrp.h"
#include "../randomforest/randomforest.h"
#include "../xgboost/xgboost.h"
#include "../secureboost/secureboost.h"
using namespace std;

struct LPMST
{
    int M = 1;
    float epsilon = 1.0;
    int seed = 0;

    RRWithPrior rrp;

    LPMST(){};
    LPMST(int M_ = 1.0, float epsilon_ = 1.0, int seed_ = 0)
    {
        M = M_;
        epsilon = epsilon_;
        seed = seed_;
    }

    void fit(XGBoostClassifier &clf, vector<XGBoostParty> &parties, vector<float> &y, vector<float> &y_hat)
    {
        _fit<XGBoostClassifier, XGBoostParty>(clf, parties, y, y_hat);
    }
    void fit(SecureBoostClassifier &clf, vector<SecureBoostParty> &parties, vector<float> &y, vector<float> &y_hat)
    {
        _fit<SecureBoostClassifier, SecureBoostParty>(clf, parties, y, y_hat);
    }
    void fit(RandomForestClassifier &clf, vector<RandomForestParty> &parties, vector<float> &y, vector<float> &y_hat)
    {
        _fit<RandomForestClassifier, RandomForestParty>(clf, parties, y, y_hat);
    }

    template <typename ModelType, typename PartyType>
    void _fit(ModelType &model, vector<PartyType> &parties, vector<float> &y, vector<float> &y_hat)
    {
        int n = y.size();
        int chunk_size = n / M;
        int class_num = *max_element(y.begin(), y.end()) + 1;
        vector<float> init_prior_dist(class_num, 1.0 / float(class_num));

        // vector<float> y_hat(chunk_size);
        int temp_ptr = 0;

        for (int m = 0; m < M; m++)
        {
            if (m == 0)
            {
                rrp = RRWithPrior(epsilon, init_prior_dist, seed);
                for (int i = temp_ptr; i < chunk_size; i++)
                {
                    y_hat.push_back(rrp.rrtop_k(y[i]));
                }
                temp_ptr = chunk_size;
            }
            else
            {
                vector<PartyType> temp_party_vec = {parties[model.active_party_id]};
                model.clear();
                model.fit(temp_party_vec, y_hat);
                for (int i = temp_ptr; i < min(n, chunk_size * (m + 1)); i++)
                {
                    vector<vector<float>> temp_x = {parties[model.active_party_id].x[i]};
                    vector<float> prior_dist = model.predict_proba(temp_x)[0];
                    rrp = RRWithPrior(epsilon, prior_dist, seed);
                    y_hat.push_back(rrp.rrtop_k(y[i]));
                }
                temp_ptr = min(n, chunk_size * (m + 1));
            }
        }

        model.clear();
        model.fit(parties, y_hat);
    }
};