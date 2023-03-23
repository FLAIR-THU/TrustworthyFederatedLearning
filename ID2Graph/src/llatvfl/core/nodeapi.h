#pragma once
#include <numeric>
#include <vector>
#include <iostream>
#include <iterator>
#include <limits>
#include <algorithm>
#include <set>
#include <tuple>
#include <queue>
using namespace std;

template <typename NodeType>
struct NodeAPI
{

    NodeAPI(){};

    /**
     * @brief Calculate the leaf purity within the specified node.
     *
     * @param node Node object
     * @param tot_cnt Total number of data samples used to train the tree.
     * @return float
     */
    float get_leaf_purity(NodeType *node, int tot_cnt)
    {
        float leaf_purity = 0;
        if (node->is_leaf_flag)
        {
            int cnt_idxs = node->idxs.size();
            if (cnt_idxs == 0)
            {
                leaf_purity = 0.0;
            }
            else
            {
                vector<int> counter(node->num_classes, 0);
                for (int i = 0; i < node->idxs.size(); i++)
                {
                    counter[node->y->at(node->idxs[i])] += 1;
                }
                leaf_purity = float(*max_element(counter.begin(), counter.end())) / float(cnt_idxs);
                leaf_purity = leaf_purity * (float(cnt_idxs) / float(tot_cnt));
            }
        }
        else
        {
            leaf_purity = get_leaf_purity(node->left, tot_cnt) + get_leaf_purity(node->right, tot_cnt);
        }
        return leaf_purity;
    }

    /**
     * @brief Returns the string informatino of the node for print-out.
     *
     * @param node Node object
     * @param show_purity Show the purity if true
     * @param binary_color Hightlight the leaf purity if true
     * @param target_party_id The party if of the attacker
     * @return string
     */
    string print(NodeType *node, bool show_purity = false, bool binary_color = true, int target_party_id = -1)
    {
        pair<string, bool> result = recursive_print(node, "", false, show_purity, binary_color, target_party_id);
        if (result.second)
        {
            return "";
        }
        else
        {
            return result.first;
        }
    }

    string to_json(NodeType *node)
    {
        string res = "";

        if (node->is_leaf_flag)
        {
            res += "{name: '*', value: " + to_string(node->record_id) +
                   ", nodeSettings: {fill: am5.color('#ADE199')}";
        }
        else
        {
            string party_name = to_string(node->party_id + 1);
            res += "{name: " + party_name +
                   ", value: " + to_string(node->record_id);

            if (node->lmir_flag_exclude_passive_parties)
            {
                res += ", nodeSettings: {fill: am5.color('#E199AD')}";
            }
            else
            {
                res += ", nodeSettings: {fill: am5.color('#67B7DC')}";
            }
        }

        if (!node->is_leaf_flag)
        {
            res += ", children: [" + to_json(node->left) + ", " + to_json(node->right) + "]";
        }
        res += "}";
        return res;
    }

    /**
     * @brief Returns the string information of the leaf node for print-out.
     *
     * @param node Node object
     * @param show_purity Show the purity if true
     * @param binary_color Hightlight the leaf purity if true
     * @return string
     */
    string print_leaf(NodeType *node, bool show_purity, bool binary_color)
    {
        string node_info = to_string(node->get_val()[0]);
        if (show_purity)
        {
            int cnt_idxs = node->idxs.size();
            if (cnt_idxs == 0)
            {
                node_info += ", null";
            }
            else
            {
                float purity = get_leaf_purity(node, node->idxs.size());
                node_info += ", ";

                if (binary_color)
                {
                    if (purity < 0.7)
                    {
                        node_info += "\033[32m";
                    }
                    else if (purity < 0.9)
                    {
                        node_info += "\033[33m";
                    }
                    else
                    {
                        node_info += "\033[31m";
                    }
                    node_info += to_string(purity);
                    node_info += "\033[0m";
                }
                else
                {
                    node_info += to_string(purity);
                }
            }
        }
        else
        {
            node_info += ", " + to_string(node->idxs.size());
        }

        return node_info;
    }

    /**
     * @brief Recusively search the nodes and returns the string information of these nodes.
     *
     * @param node
     * @param prefix
     * @param isleft
     * @param show_purity
     * @param binary_color
     * @param target_party_id
     * @return pair<string, bool>
     */
    pair<string, bool> recursive_print(NodeType *node, string prefix, bool isleft, bool show_purity,
                                       bool binary_color, int target_party_id = -1)
    {
        string node_info;
        bool skip_flag;
        if (node->is_leaf_flag)
        {
            skip_flag = node->depth <= 0 && target_party_id != -1 && node->party_id != target_party_id;
            if (skip_flag)
            {
                node_info = "";
            }
            else
            {
                node_info = print_leaf(node, show_purity, binary_color);
            }
            node_info = prefix + "|-- " + node_info;
            node_info += "\n";
        }
        else
        {
            node_info += to_string(node->get_party_id());
            node_info += ", ";
            node_info += to_string(node->get_record_id());
            if (node->lmir_flag_exclude_passive_parties)
            {
                node_info += " *";
            }
            node_info = prefix + "|-- " + node_info;

            string next_prefix = "";
            if (isleft)
            {
                next_prefix += "|    ";
            }
            else
            {
                next_prefix += "     ";
            }

            pair<string, bool> left_node_info_and_skip_flag = recursive_print(node->left, prefix + next_prefix, true,
                                                                              show_purity, binary_color, target_party_id);
            pair<string, bool> right_node_info_and_skip_flag = recursive_print(node->right, prefix + next_prefix, false,
                                                                               show_purity, binary_color, target_party_id);
            if (left_node_info_and_skip_flag.second && right_node_info_and_skip_flag.second)
            {
                node_info += " -> " + print_leaf(node, show_purity, binary_color);
                node_info += "\n";
            }
            else
            {
                node_info += "\n";
                node_info += left_node_info_and_skip_flag.first;
                node_info += right_node_info_and_skip_flag.first;
            }

            skip_flag = false;
        }

        return make_pair(node_info, skip_flag);
    }

    /**
     * @brief Returns the prediction for a single data point.
     *
     * @param node
     * @param xi
     * @return vector<float>
     */
    vector<float> predict_row(NodeType *node, vector<float> &xi)
    {
        queue<NodeType *> que;
        que.push(node);

        NodeType *temp_node;
        while (!que.empty())
        {
            temp_node = que.front();
            que.pop();

            if (temp_node->is_leaf_flag)
            {
                return temp_node->val;
            }
            else
            {
                if (node->parties->at(temp_node->party_id).is_left(temp_node->record_id, xi))
                {
                    que.push(temp_node->left);
                }
                else
                {
                    que.push(temp_node->right);
                }
            }
        }

        vector<float> nan_vec(node->num_classes, nan(""));
        return nan_vec;
    }

    /**
     * @brief Returns the prediction for the batched data samples.
     *
     * @param node
     * @param x_new
     * @return vector<vector<float>>
     */
    vector<vector<float>> predict(NodeType *node, vector<vector<float>> &x_new)
    {
        int x_new_size = x_new.size();
        vector<vector<float>> y_pred(x_new_size);
        for (int i = 0; i < x_new_size; i++)
        {
            y_pred[i] = predict_row(node, x_new[i]);
        }
        return y_pred;
    }
};