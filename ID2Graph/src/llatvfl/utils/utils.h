#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
using namespace std;

/**
 * @brief Sigmoid function
 *
 * @param x
 * @return float
 */
inline float sigmoid(float x)
{
    float sigmoid_range = 34.538776394910684;
    if (x <= -1 * sigmoid_range)
        return 1e-15;
    else if (x >= sigmoid_range)
        return 1.0 - 1e-15;
    else
        return 1.0 / (1.0 + exp(-1 * x));
}

/**
 * @brief Softmax function
 *
 * @param x
 * @return vector<float>
 */
inline vector<float> softmax(vector<float> x)
{
    int n = x.size();
    float max_x = *max_element(x.begin(), x.end());
    vector<float> numerator(n, 0);
    vector<float> output(n, 0);
    float denominator = 0;

    for (int i = 0; i < n; i++)
    {
        numerator[i] = exp(x[i] - max_x);
        denominator += numerator[i];
    }

    for (int i = 0; i < n; i++)
    {
        output[i] = numerator[i] / denominator;
    }

    return output;
}

/**
 * @brief Return unique elements of the given vector
 *
 * @tparam T
 * @param inData
 * @return vector<T>
 */
template <typename T>
inline vector<T> remove_duplicates(vector<T> &inData)
{
    vector<float> outData;
    set<float> s{};
    for (int i = 0; i < inData.size(); i++)
    {
        if (s.insert(inData[i]).second)
        {
            outData.push_back(inData[i]);
        }
    }
    return outData;
}

template <typename T>
static inline float Lerp(T v0, T v1, T t)
{
    return (1 - t) * v0 + t * v1;
}

/**
 * @brief Return quantiles
 *
 * @tparam T
 * @param inData
 * @param probs
 * @return std::vector<T>
 */
template <typename T>
static inline std::vector<T> Quantile(const std::vector<T> &inData, const std::vector<T> &probs)
{
    if (inData.empty())
    {
        return std::vector<T>();
    }

    if (1 == inData.size())
    {
        return std::vector<T>(1, inData[0]);
    }

    std::vector<T> data = inData;
    std::sort(data.begin(), data.end());
    std::vector<T> quantiles;

    for (size_t i = 0; i < probs.size(); ++i)
    {
        T poi = Lerp<T>(-0.5, data.size() - 0.5, probs[i]);

        size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
        size_t right = std::min(int64_t(std::ceil(poi)), int64_t(data.size() - 1));

        T datLeft = data.at(left);
        T datRight = data.at(right);

        T quantile = Lerp<T>(datLeft, datRight, poi - left);

        quantiles.push_back(quantile);
    }

    return quantiles;
}

inline vector<int> get_num_parties_per_process(int n_job, int num_parties)
{
    vector<int> num_parties_per_thread(n_job, num_parties / n_job);
    for (int i = 0; i < num_parties % n_job; i++)
    {
        num_parties_per_thread[i] += 1;
    }
    return num_parties_per_thread;
}

inline bool is_satisfied_with_lmir_bound(int num_classes, float xi,
                                         vector<float> &y,
                                         vector<float> &entire_class_cnt,
                                         vector<float> &prior,
                                         vector<int> &idxs_within_node)
{
    float eps = 1e-15;

    if (xi > 0)
    {
        int num_row = y.size();
        int num_idxs_within_node = idxs_within_node.size();

        vector<float> y_class_cnt_within_node(num_classes, 0);
        for (int j = 0; j < num_idxs_within_node; j++)
        {
            y_class_cnt_within_node[int(y[idxs_within_node[j]])] += 1;
        }

        float in_kl_divergence = 0;
        float out_kl_divergence = 0;

        float nc_div_n;
        float Nc_div_N;
        float Nc_m_nc_div_N_m_n;

        for (int c = 0; c < num_classes; c++)
        {
            nc_div_n = y_class_cnt_within_node[c] / float(num_idxs_within_node);
            Nc_div_N = prior[c];
            Nc_m_nc_div_N_m_n = (entire_class_cnt[c] - y_class_cnt_within_node[c]) / float(num_row - num_idxs_within_node);

            in_kl_divergence += nc_div_n * log(eps + nc_div_n / Nc_div_N);
            out_kl_divergence += Nc_m_nc_div_N_m_n * log(eps + Nc_m_nc_div_N_m_n / Nc_div_N);
        }

        return max(in_kl_divergence, out_kl_divergence) <= xi;
    }
    else
    {
        return true;
    }
}

inline bool is_satisfied_with_lmir_bound_from_pointer(int num_classes, float xi,
                                                      vector<float> *y,
                                                      vector<float> &entire_class_cnt,
                                                      vector<float> *prior,
                                                      vector<int> &idxs_within_node)
{
    float eps = 1e-15;

    if (xi > 0)
    {
        int num_row = y->size();
        int num_idxs_within_node = idxs_within_node.size();

        vector<float> y_class_cnt_within_node(num_classes, 0);
        for (int j = 0; j < num_idxs_within_node; j++)
        {
            y_class_cnt_within_node[int(y->at(idxs_within_node[j]))] += 1;
        }

        float in_kl_divergence = 0;
        float out_kl_divergence = 0;

        float nc_div_n;
        float Nc_div_N;
        float Nc_m_nc_div_N_m_n;

        for (int c = 0; c < num_classes; c++)
        {
            nc_div_n = y_class_cnt_within_node[c] / float(num_idxs_within_node);
            Nc_div_N = prior->at(c);
            Nc_m_nc_div_N_m_n = (entire_class_cnt[c] - y_class_cnt_within_node[c]) / float(num_row - num_idxs_within_node);

            in_kl_divergence += nc_div_n * log(eps + nc_div_n / Nc_div_N);
            out_kl_divergence += Nc_m_nc_div_N_m_n * log(eps + Nc_m_nc_div_N_m_n / Nc_div_N);
        }

        return max(in_kl_divergence, out_kl_divergence) <= xi;
    }
    else
    {
        return true;
    }
}