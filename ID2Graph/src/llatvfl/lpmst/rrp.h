#pragma once
#include <map>
#include <random>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
using namespace std;

struct RRWithPrior
{
    vector<float> prior_dist;
    vector<int> sort_idx; // k-th value is top-k label
    unordered_map<int, int> label2argmaxpos;
    float epsilon;
    int K;

    int k_star;
    float best_w_k;
    float threshold_prob;

    mt19937 gen;
    uniform_real_distribution<> uniform_dist_0_to_1;
    uniform_int_distribution<> uniform_dist_0_to_Km1;
    uniform_int_distribution<> uniform_dist_1_to_kstarm1;

    RRWithPrior(){};
    RRWithPrior(float epsilon_, int K_, int seed = 0)
    {
        epsilon = epsilon_;
        K = K_;
        prior_dist = vector<float>(K, 1.0 / float(K));

        _set_random(seed);
        _set_label2argmaxpos();
        search_optimal_k();
    }
    RRWithPrior(float epsilon_, vector<float> &prior_dist_, int seed = 0)
    {
        epsilon = epsilon_;
        K = prior_dist_.size();
        prior_dist = prior_dist_;

        _set_random(seed);
        _set_label2argmaxpos();
        search_optimal_k();
    }

    void _set_random(int seed)
    {
        gen = mt19937(seed);
        srand(seed);
        uniform_dist_0_to_1 = uniform_real_distribution<>(0, 1);
        uniform_dist_0_to_Km1 = uniform_int_distribution<>(0, K - 1);
    }

    void _set_label2argmaxpos()
    {
        sort_idx.resize(K);
        iota(sort_idx.begin(), sort_idx.end(), 0);
        stable_sort(sort_idx.begin(), sort_idx.end(),
                    [this](size_t i1, size_t i2)
                    { return this->prior_dist[i1] > this->prior_dist[i2]; });
        for (int k = 0; k < K; k++)
        {
            label2argmaxpos.emplace(make_pair(sort_idx[k], k));
        }
    }

    void search_optimal_k()
    {
        float temp_w_k;
        float cumulative_p = 0;
        float max_w_k = 0;
        float exp_eps = exp(epsilon);
        for (int k = 0; k < K; k++)
        {
            cumulative_p += prior_dist[sort_idx[k]];
            temp_w_k = exp_eps / (exp_eps + float(k)) * cumulative_p;

            if (temp_w_k > max_w_k)
            {
                max_w_k = temp_w_k;
                k_star = k + 1;
            }
        }

        best_w_k = max_w_k;
        threshold_prob = exp_eps / (exp_eps + float(k_star) - 1);
        uniform_dist_1_to_kstarm1 = uniform_int_distribution<>(1, k_star - 1);
    }

    int rrtop_k(int y)
    {
        int y_random;
        int temp_idx;
        if (label2argmaxpos[y] < k_star)
        {
            if (threshold_prob > uniform_dist_0_to_1(gen))
            {
                y_random = y;
            }
            else
            {
                temp_idx = uniform_dist_1_to_kstarm1(gen);
                y_random = sort_idx[temp_idx - 1];
                if (y_random == y)
                {
                    y_random = sort_idx[k_star - 1];
                }
            }
        }
        else
        {
            y_random = uniform_dist_0_to_Km1(gen);
        }

        return y_random;
    }
};