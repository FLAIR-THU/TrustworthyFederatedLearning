import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class BiInteractionPooling(nn.Module):
    """Bi-Interaction Layer used in Neural FM,compress the
     pairwise element-wise product of features into one single vector.

      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      References
        - [He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](http://arxiv.org/abs/1708.05027)
    """

    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = torch.pow(
            torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(
            concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term


class SENETLayer(nn.Module):
    """SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size,embedding_size)``.
      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, reduction_ratio=3,  seed=1024, device='cpu'):
        super(SENETLayer, self).__init__()
        self.seed = seed
        self.filed_size = filed_size
        self.reduction_size = max(1, filed_size // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(self.filed_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.filed_size, bias=False),
            nn.ReLU()
        )
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        Z = torch.mean(inputs, dim=-1, out=None)
        A = self.excitation(Z)
        V = torch.mul(inputs, torch.unsqueeze(A, dim=2))

        return V


class BilinearInteraction(nn.Module):
    """BilinearInteraction Layer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,filed_size, embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size, embedding_size)``.
      Arguments
        - **str** : String, types of bilinear functions used in this layer.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
Tongwen](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, filed_size, embedding_size, bilinear_type="interaction", seed=1024, device='cpu'):
        super(BilinearInteraction, self).__init__()
        self.bilinear_type = bilinear_type
        self.seed = seed
        self.bilinear = nn.ModuleList()
        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(embedding_size, embedding_size, bias=False)
        elif self.bilinear_type == "each":
            for i in range(filed_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        elif self.bilinear_type == "interaction":
            for i, j in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == "each":
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == "interaction":
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)


class CIN(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        - **seed** : A Python integer to use as random seed.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, field_nums, layer_size=(128, 128), activation=F.relu, split_half=True, l2_reg=1e-5, seed=1024, device='cpu'):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = layer_size
        self.field_nums = [field_nums]
        self.split_half = split_half
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed

        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

    #         for tensor in self.conv1ds:
    #             nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            # x.shape = (batch_size , hi * m, dim)
            x = x.reshape(
                batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            # x.shape = (batch_size , hi, dim)
            x = self.conv1ds[i](x)

            if self.activation is None or self.activation=='linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)

        return result


class AFMLayer(nn.Module):
    """Attentonal Factorization Machine models pairwise (order-2) feature
    interactions without linear term and bias.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      Arguments
        - **attention_factor** : Positive integer, dimensionality of the
         attention network output space.
        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength
         applied to attention network.
        - **dropout_rate** : float between in [0,1). Fraction of the attention net output units to dropout.
        - **seed** : A Python integer to use as random seed.
      References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(self, in_feature, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, device='cpu'):
        super(AFMLayer, self).__init__()
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        embedding_size = in_feature

        self.attention_W = nn.Parameter(torch.Tensor(
            embedding_size, self.attention_factor))

        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))

        self.projection_h = nn.Parameter(
            torch.Tensor(self.attention_factor, 1))

        self.projection_p = nn.Parameter(torch.Tensor(embedding_size, 1))

        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor, )

        self.dropout = nn.Dropout(dropout_rate)

        self.to(device)

    def forward(self, inputs):
        embeds_vec_list = inputs
        row = []
        col = []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = torch.cat(row, dim=1)
        q = torch.cat(col, dim=1)
        inner_product = p * q

        bi_interaction = inner_product
        attention_temp = F.relu(torch.tensordot(
            bi_interaction, self.attention_W, dims=([-1], [0])) + self.attention_b)

        self.normalized_att_score = F.softmax(torch.tensordot(
            attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        attention_output = torch.sum(
            self.normalized_att_score * bi_interaction, dim=1)

        attention_output = self.dropout(attention_output)  # training

        afm_out = torch.tensordot(
            attention_output, self.projection_p, dims=([-1], [0]))
        return afm_out


class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,att_embedding_size * head_num)``.
      Arguments
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, in_feature, att_embedding_size=8, head_num=2, use_res=True, seed=1024, device='cpu'):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed

        embedding_size = in_feature

        self.W_Query = nn.Parameter(torch.Tensor(
            embedding_size, self.att_embedding_size * self.head_num))

        self.W_key = nn.Parameter(torch.Tensor(
            embedding_size, self.att_embedding_size * self.head_num))

        self.W_Value = nn.Parameter(torch.Tensor(
            embedding_size, self.att_embedding_size * self.head_num))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(
                embedding_size, self.att_embedding_size * self.head_num))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        querys = torch.tensordot(inputs, self.W_Query,
                                 dims=([-1], [0]))  # None F D*head_num
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D

        querys = torch.stack(torch.split(
            querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(
            values, self.att_embedding_size, dim=2))
        inner_product = torch.einsum(
            'bnik,bnjk->bnij', querys, keys)  # head_num None F F

        self.normalized_att_scores = F.softmax(
            inner_product, dim=1)  # head_num None F F
        result = torch.matmul(self.normalized_att_scores,
                              values)  # head_num None F D

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D*head_num
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)

        return result


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, input_feature_num, layer_num=2, seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(input_feature_num, 1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(input_feature_num, 1))) for i in range(self.layer_num)])
        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l