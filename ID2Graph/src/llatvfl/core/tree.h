#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "../core/nodeapi.h"
using namespace std;

template <typename NodeType>
struct Tree
{
    NodeType dtree;
    NodeAPI<NodeType> nodeapi;
    int num_row;

    Tree() {}

    /**
     * @brief Get the root node object
     *
     * @return NodeType&
     */
    NodeType &get_root_node()
    {
        return *dtree;
    }

    /**
     * @brief Return the prediction of the given data
     *
     * @param X
     * @return vector<vector<float>>
     */
    vector<vector<float>> predict(vector<vector<float>> &X)
    {
        return nodeapi.predict(&dtree, X);
    }

    /**
     * @brief Return the predictions of the training data assigned to the node
     *
     * @param node
     * @return vector<pair<vector<int>, vector<vector<float>>>>
     */
    vector<pair<vector<int>, vector<vector<float>>>> extract_train_prediction_from_node(NodeType &node)
    {
        if (node.is_leaf_flag)
        {
            vector<pair<vector<int>, vector<vector<float>>>> result;
            result.push_back(make_pair(node.idxs,
                                       vector<vector<float>>(node.idxs.size(),
                                                             node.val)));
            return result;
        }
        else
        {
            vector<pair<vector<int>, vector<vector<float>>>> left_result =
                extract_train_prediction_from_node(*node.left);
            vector<pair<vector<int>, vector<vector<float>>>> right_result =
                extract_train_prediction_from_node(*node.right);
            left_result.insert(left_result.end(), right_result.begin(), right_result.end());
            return left_result;
        }
    }

    /**
     * @brief Returns the predictions of the training dataset
     *
     * @return vector<vector<float>>
     */
    vector<vector<float>> get_train_prediction()
    {
        vector<pair<vector<int>, vector<vector<float>>>> result = extract_train_prediction_from_node(dtree);
        vector<vector<float>> y_train_pred(num_row);
        for (int i = 0; i < result.size(); i++)
        {
            for (int j = 0; j < result[i].first.size(); j++)
            {
                y_train_pred[result[i].first[j]] = result[i].second[j];
            }
        }

        return y_train_pred;
    }

    string to_json()
    {
        return nodeapi.to_json(&dtree);
    }

    string to_html()
    {
        string html_literal_1 = "<!-- Styles -->\n<style>\n    "
                                "body {\n        font-family: -apple-system, BlinkMacSystemFont, "
                                "'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', "
                                "'Segoe UI Emoji', 'Segoe UI Symbol';\n    }\n\n    "
                                "#chartdiv {\n        width: 100%;\n        height: 400px;\n    }\n"
                                "</style>\n\n<!-- Resources -->\n<script src='https://cdn.amcharts.com/lib/5/index.js'></script>\n"
                                "<script src='https://cdn.amcharts.com/lib/5/xy.js'></script>\n"
                                "<script src='https://cdn.amcharts.com/lib/5/hierarchy.js'></script>\n"
                                "<script src='https://cdn.amcharts.com/lib/5/themes/Animated.js'></script>\n"
                                "<script src='https://cdn.amcharts.com/lib/5/plugins/exporting.js'></script>\n\n"
                                "<!-- Chart code -->\n<script>\n    am5.ready(function () "
                                "{\n\n\n        /**\n         * ---------------------------------------\n         "
                                "* This demo was created using amCharts 5.\n         *\n         "
                                "* For more information visit:\n         * https://www.amcharts.com/\n         *\n         "
                                "* Documentation is available at:\n         * https://www.amcharts.com/docs/v5/\n         "
                                "* ---------------------------------------\n         */\n\n        // Create root and chart\n        "
                                "var root = am5.Root.new('chartdiv');\n\n        "
                                "root.setThemes([\n            am5themes_Animated.new(root)\n        ]);\n\n        "
                                "var data = [";
        string html_literal_2 = "];\n\n        "
                                "var container = root.container.children.push(\n            "
                                "am5.Container.new(root, {\n                "
                                "width: am5.percent(100),\n                "
                                "height: am5.percent(100),\n                "
                                "layout: root.verticalLayout\n            })\n        );\n\n        "
                                "var series = container.children.push(\n            "
                                "am5hierarchy.Tree.new(root, {\n                "
                                "singleBranchOnly: false,\n                downDepth: 1,\n                "
                                "initialDepth: 5,\n                topDepth: 0,\n                "
                                "valueField: 'value',\n                categoryField: 'name',\n                "
                                "childDataField: 'children',\n                "
                                "orientation: 'vertical'\n            })\n        );\n\n        "
                                "series.circles.template.setAll({templateField: 'nodeSettings'});\n"
                                "series.data.setAll(data);\n        "
                                "series.set('selectedDataItem', series.dataItems[0]);\n\n\n        "
                                "var exporting = am5plugins_exporting.Exporting.new(root, {\n            "
                                "menu: am5plugins_exporting.ExportingMenu.new(root, {}),\n        });\n    });"
                                "\n</script>\n\n<!-- HTML -->\n<div id='chartdiv'></div>";

        return html_literal_1 + to_json() + html_literal_2;
    }

    /**
     * @brief Print out the structure of this tree
     *
     * @param show_purity show purity if true
     * @param binary_color color each node if true
     * @param target_party_id
     * @return string
     */
    string print(bool show_purity = false, bool binary_color = true, int target_party_id = -1)
    {
        return nodeapi.print(&dtree, show_purity, binary_color, target_party_id);
    }

    /**
     * @brief Get the average leaf purity of this tree
     *
     * @return float
     */
    float get_leaf_purity()
    {
        return nodeapi.get_leaf_purity(&dtree, num_row);
    }
};
