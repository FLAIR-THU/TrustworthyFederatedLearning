# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""
from __future__ import print_function

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .inputs import build_input_features, SparseFeat, DenseFeat
from .layers import PredictionLayer
from .layers.utils import slice_arrays


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = self.create_embedding_matrix(self.sparse_feature_columns, 1, init_std, sparse=False).to(
            device)

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(len(self.dense_feature_columns), 1)).to(
                device)
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            linear_sparse_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
            linear_dense_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit = linear_sparse_logit + linear_dense_logit
        elif len(sparse_embedding_list) > 0:
            linear_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
        elif len(dense_value_list) > 0:
            linear_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
        else:
            linear_logit = torch.zeros([X.shape[0],1])
        return linear_logit

    def create_embedding_matrix(self, feature_columns, embedding_size, init_std=0.0001, sparse=False):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        embedding_dict = nn.ModuleDict(
            {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=sparse) for feat in
             sparse_feature_columns}
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        return embedding_dict


class BaseModel(nn.Module):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128),
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu'):

        super(BaseModel, self).__init__()

        self.reg_loss = torch.zeros((1,), device=device)
        self.device = device  # device

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = self.create_embedding_matrix(dnn_feature_columns, embedding_size, init_std,
                                                           sparse=False).to(device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.add_regularization_loss(
            self.embedding_dict.parameters(), l2_reg_embedding)
        self.add_regularization_loss(
            self.linear_model.parameters(), l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

    def fit(self, x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            initial_epoch=0,
            validation_split=0.,
            validation_data=None,
            shuffle=True, ):
        if validation_data:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)

        elif validation_split and 0. < validation_split < 1.:
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.hstack(list(map(lambda x: np.expand_dims(x, axis=1), x)))),
            torch.from_numpy(y))

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        print(self.device, end="\n")
        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        print("Train on {0} samples, validate on {1} samples".format(
            len(train_tensor_data), len(val_y)))
        for epoch in range(initial_epoch, epochs):
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            # if abs(loss_last - loss_now) < 0.0
            sample_num = len(train_tensor_data)
            train_result = {}
            steps_per_epoch = (sample_num - 1) // batch_size + 1
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for index, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')

                        total_loss = loss + self.reg_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward(retain_graph=True)
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy()))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            epoch_time = int(time.time() - start_time)
            if verbose > 0:
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, total_loss_epoch / sample_num)

                for name, result in train_result.items():
                    eval_str += " - " + name + \
                        ": {0: .4f}".format(np.sum(result) / steps_per_epoch)

                if len(val_x) and len(val_y):
                    eval_result = self.evaluate(val_x, val_y, batch_size)

                    for name, result in eval_result.items():
                        eval_str += " - val_" + name + \
                            ": {0: .4f}".format(result)
                print(eval_str)

    def evaluate(self, x, y, batch_size=256):
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):

        model = self.eval()
        x = np.hstack(list(map(lambda x: np.expand_dims(x, axis=1), x)))
        tensor_data = Data.TensorDataset(torch.from_numpy(x))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for index, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                # y = y_test.to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)
        return np.concatenate(pred_ans)

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list, dense_value_list

    def create_embedding_matrix(self, feature_columns, embedding_size, init_std=0.0001, sparse=False):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        embedding_dict = nn.ModuleDict(
            {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=sparse) for feat in
             sparse_feature_columns}
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        return embedding_dict

    def compute_input_dim(self, feature_columns, embedding_size, dense_only=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        if dense_only:
            return sum(map(lambda x: x.dimension, dense_feature_columns))
        else:

            return len(sparse_feature_columns) * embedding_size + sum(map(lambda x: x.dimension, dense_feature_columns))

    def add_regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = torch.zeros((1,), device=self.device)
        for w in weight_list:
            if isinstance(w, tuple):
                l2_reg = torch.norm(w[1], p=p, )
            else:
                l2_reg = torch.norm(w, p=p, )
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        self.reg_loss += reg_loss

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None):

        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _get_metrics(self, metrics):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
        return metrics_
