#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <cmath>
#include "../core/model.h"
#include "../utils/utils.h"
#include "tree.h"
using namespace std;

/**
 * @brief Base struture for loss functions
 *
 */
struct LossFunc
{
    LossFunc(){};

    /**
     * @brief Get the aggregated loss value
     *
     * @param y_pred predicted value
     * @param y ground-truth
     * @return float
     */
    virtual float get_loss(vector<vector<float>> &y_pred, vector<float> &y) = 0;

    /**
     * @brief Get the gradient of the loss w.r.t the predicted value
     *
     * @param y_pred
     * @param y
     * @return vector<vector<float>>
     */
    virtual vector<vector<float>> get_grad(vector<vector<float>> &y_pred, vector<float> &y) = 0;

    /**
     * @brief Get the hessian of the loss w.r.t the predicted value
     *
     * @param y_pred
     * @param y
     * @return vector<vector<float>>
     */
    virtual vector<vector<float>> get_hess(vector<vector<float>> &y_pred, vector<float> &y) = 0;
};

/**
 * @brief Implementation of Binary Closs Entropy Loss
 *
 */
struct BCELoss : LossFunc
{
    BCELoss(){};

    /**
     * @brief Get the averaged binacy cross entropy loss
     *
     * @param y_pred predicted value
     * @param y ground-truth
     * @return float
     */
    float get_loss(vector<vector<float>> &y_pred, vector<float> &y)
    {
        float loss = 0;
        float n = y_pred.size();
        for (int i = 0; i < n; i++)
        {
            if (y[i] == 1)
            {
                loss += log(1 + exp(-1 * sigmoid(y_pred[i][0]))) / n;
            }
            else
            {
                loss += log(1 + exp(sigmoid(y_pred[i][0]))) / n;
            }
        }
        return loss;
    }

    /**
     * @brief Get the gradient of the loss w.r.t the predicted value
     *
     * @param y_pred
     * @param y
     * @return vector<vector<float>>
     */
    vector<vector<float>> get_grad(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int element_num = y_pred.size();
        vector<vector<float>> grad(element_num);
        for (int i = 0; i < element_num; i++)
            grad[i] = {sigmoid(y_pred[i][0]) - y[i]};
        return grad;
    }

    /**
     * @brief Get the hessian of the loss w.r.t the predicted value
     *
     * @param y_pred
     * @param y
     * @return vector<vector<float>>
     */
    vector<vector<float>> get_hess(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int element_num = y_pred.size();
        vector<vector<float>> hess(element_num);
        for (int i = 0; i < element_num; i++)
        {
            float temp_proba = sigmoid(y_pred[i][0]);
            hess[i] = {temp_proba * (1 - temp_proba)};
        }
        return hess;
    }
};

/**
 * @brief Implementation of Cross Entropy Loss
 *
 */
struct CELoss : LossFunc
{
    int num_classes;

    CELoss(){};
    CELoss(int num_classes_) { num_classes = num_classes_; }

    /**
     * @brief Get the averaged cross entropy loss
     *
     * @param y_pred predicted value
     * @param y ground-truth
     * @return float
     */
    float get_loss(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int n = y_pred.size();

        vector<vector<float>> y_pred_proba(n);
        for (int i = 0; i < n; i++)
        {
            y_pred_proba[i] = softmax(y_pred[i]);
        }

        float loss = 0;
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < num_classes; c++)
            {
                if (y[i] == c)
                {
                    loss -= log(y_pred_proba[i][c]);
                }
            }
        }
        return loss;
    }

    /**
     * @brief Get the gradient of the loss w.r.t the predicted value
     *
     * @param y_pred
     * @param y
     * @return vector<vector<float>>
     */
    vector<vector<float>> get_grad(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int n = y_pred.size();

        vector<vector<float>> y_pred_proba(n);
        for (int i = 0; i < n; i++)
        {
            y_pred_proba[i] = softmax(y_pred[i]);
        }

        vector<vector<float>> grad(n, vector<float>(num_classes, 0));

        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < num_classes; c++)
            {
                grad[i][c] = y_pred_proba[i][c];
                if (y[i] == c)
                {
                    grad[i][c] -= 1;
                }
            }
        }
        return grad;
    }

    /**
     * @brief Get the hessian of the loss w.r.t the predicted value
     *
     * @param y_pred
     * @param y
     * @return vector<vector<float>>
     */
    vector<vector<float>> get_hess(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int n = y_pred.size();

        vector<vector<float>> y_pred_proba(n);
        for (int i = 0; i < n; i++)
        {
            y_pred_proba[i] = softmax(y_pred[i]);
        }

        vector<vector<float>> hess(n, vector<float>(num_classes, 0));

        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < num_classes; c++)
            {
                hess[i][c] = y_pred_proba[i][c] * (1 - y_pred_proba[i][c]);
            }
        }
        return hess;
    }
};
