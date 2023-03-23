#pragma once
#include <iostream>
#include <limits>
#include <vector>
#include "../utils/dok.h"
using namespace std;

struct Graph
{
    unsigned long num_nodes = 0;
    unsigned long num_links = 0;
    float total_weight = 0;

    // cumulative degree for each node, deg(0) = degrees[0]
    // deg(k) = degrees[k] - degrees[k-1]
    vector<unsigned long> degrees;
    vector<vector<int>> nodes; // current node idx to original record idxs
    vector<unsigned int> links;
    vector<float> weights; // TODO check if `float` works or not

    Graph(){};
    Graph(vector<vector<int>> &node2original_records)
    {
        num_nodes = 0;
        num_links = 0;
        total_weight = 0;

        int size_of_node2original_records = node2original_records.size();
        nodes.reserve(size_of_node2original_records);
        for (size_t i = 0; i < size_of_node2original_records; i++)
        {
            nodes.push_back(node2original_records[i]);
        }
    }
    Graph(SparseMatrixDOK<float> &sm_dok)
    {
        num_nodes = sm_dok.dim_row;
        // initilize first graph without contraction
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            vector<int> n;
            n.push_back(i);
            nodes.push_back(n);
        }

        int temp_num_links = 0;
        int cum_degrees = 0;
        degrees.resize(num_nodes, 0);
        for (int i = 0; i < num_nodes; i++)
        {
            temp_num_links = sm_dok.row2nonzero_idx[i].size();
            cum_degrees += temp_num_links;
            degrees[i] = cum_degrees;

            for (int j = 0; j < temp_num_links; j++)
            {
                links.push_back(sm_dok.row2nonzero_idx[i][j]);
                weights.push_back(sm_dok(i, sm_dok.row2nonzero_idx[i][j]));
            }

            sm_dok.row2nonzero_idx[i].clear();
            sm_dok.row2nonzero_idx[i].shrink_to_fit();
        }

        sm_dok.row2nonzero_idx.clear();
        sm_dok.row2nonzero_idx.shrink_to_fit();
        sm_dok.um_ij2w.clear();

        // compute total weight
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            total_weight += (float)get_weighted_degree(i);
        }
    }
    Graph(unsigned long num_nodes_, vector<unsigned long> &degrees_,
          vector<unsigned int> &links_, vector<float> &weights_)
    {
        num_nodes = num_nodes_;
        degrees = degrees_;
        links = links_;
        weights = weights_;

        // initilize first graph without contraction
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            vector<int> n;
            n.push_back(i);
            nodes.push_back(n);
        }

        // compute total weight
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            total_weight += (float)get_weighted_degree(i);
        }
    }

    /**
     * @brief Add a node
     *
     * @param n
     */
    void add_nose(vector<int> &n)
    {
        nodes.push_back(n);
    }

    /**
     * @brief Get pointers to the first neighbor and weight of the edge between the node and the neighbor
     *
     * @param node
     * @return pair<vector<unsigned int>::iterator, vector<float>::iterator>
     */
    pair<vector<unsigned int>::iterator, vector<float>::iterator> get_neighbors(unsigned int node)
    {
        if (node == 0)
        {
            return make_pair(links.begin(), weights.begin());
        }
        else if (weights.size() != 0)
        {
            return make_pair(links.begin() + degrees[node - 1], weights.begin() + degrees[node - 1]);
        }
        else
        {
            return make_pair(links.begin() + degrees[node - 1], weights.begin());
        }
    }

    /**
     * @brief Get the number of neighbors of the node
     *
     * @param node
     * @return unsigned int
     */
    unsigned int get_num_neighbors(unsigned int node)
    {
        if (node == 0)
        {
            return degrees[0];
        }
        else
        {
            return degrees[node] - degrees[node - 1];
        }
    }

    /**
     * @brief Get the number or the weight of self loops of the node
     *
     * @param node
     * @return float
     */
    float get_num_selfloops(unsigned int node)
    {
        bool is_weights_size_is_not_zero = weights.size() != 0;
        pair<vector<unsigned int>::iterator, vector<float>::iterator> p = get_neighbors(node);
        unsigned int num_neighbors = get_num_neighbors(node);
        for (unsigned int i = 0; i < num_neighbors; i++)
        {
            if (*(p.first + i) == node)
            {
                if (is_weights_size_is_not_zero)
                {
                    return (float)*(p.second + i);
                }
                else
                {
                    return 1;
                }
            }
        }
        return 0.;
    }

    /**
     * @brief Get the weighed degree of the node
     *
     * @param node
     * @return float
     */
    float get_weighted_degree(unsigned int node)
    {
        if (weights.size() == 0)
        {
            return (float)get_num_neighbors(node);
        }
        else
        {
            unsigned int num_neighbors = get_num_neighbors(node);
            pair<vector<unsigned int>::iterator, vector<float>::iterator> p = get_neighbors(node);
            float res = 0;
            for (unsigned int i = 0; i < num_neighbors; i++)
            {
                res += (float)*(p.second + i);
            }
            return res;
        }
    }
};