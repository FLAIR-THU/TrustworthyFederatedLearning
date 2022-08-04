import numpy as np
import torch
import sys, os

sys.path.append(os.pardir)
from heat_map import show_heat_map
from models.autoencoder import AutoEncoder
from utils import label_to_onehot, calculate_entropy


def show_autoencoder_transform_result(autoencoder, num_classes):
    labels = torch.tensor(np.arange(0, num_classes))
    one_hot_label = label_to_onehot(labels, num_classes)

    print(f"one_hot_label: \n{one_hot_label}")

    dec_label, enc_prob = autoencoder(one_hot_label)
    entropy_value = calculate_entropy(enc_prob, N=num_classes)
    print("enc_prob: \n {}".format(np.round(enc_prob.detach().numpy(), 2)))
    print("enc_entropy_value : {}".format(entropy_value))
    print("dec_label: \n {}".format(np.round(dec_label.detach().numpy(), 2)))
    return dec_label, enc_prob


if __name__ == '__main__':
    device = "cpu"
    n_classes = 10
    dim = n_classes

    # model_timestamp = '1630382344'  # n_classes 10 no entropy loss
    # model_timestamp = '1629006257' # n_classes 2
    # model_timestamp = '1629007287'  # n_classes 4
    # model_timestamp = '1629007376'  # n_classes 6
    # model_timestamp = "1629011069" # n_classes 8
    # model_timestamp = "1628350565"  # n_classes 10
    # model_timestamp = "1629187329"  # n_classes 10
    # model_timestamp = "1629302511"  # n_classes 10
    # model_timestamp = '1629787011'  # n_classes 10
    # model_timestamp = '1629828593'  # n_classes 10
    # model_timestamp = '1629830931'  # n_classes 10 *

    # model_timestamp = '1630634934' # n_classes 10
    # model_timestamp = '1630637969'  # n_classes 10 lba 1.5
    # lda = 1.5
    # lda = 1.0
    # model_timestamp = '1630557642'  # n_classes 20
    # model_timestamp = '1630308532'  # n_classes 5
    # model_timestamp = '1630895642'  # n_classes 5
    # lda = 0.1
    # model_timestamp = '1630981788'  # n_classes 5
    # lda = 0.1
    # model_timestamp = '1630879452'  # n_classes 5
    # lda = 1.0
    # model_timestamp = '1630891447'  # n_classes 5
    # lda = 0.5

    # model_timestamp = '1631059899'  # n_classes 5
    # model_timestamp = '1631060978'  # n_classes 10
    model_timestamp = '1631093149'
    lda = 0.0
    encoder = AutoEncoder(input_dim=dim, encode_dim=2 + dim * 6).to(device)
    # encoder = AutoEncoder(input_dim=dim, encode_dim=2 + i * 10).to(device)
    model_name = f"autoencoder_{dim}_{lda}_{model_timestamp}"
    # model_name = f"autoencoder_{dim}_{model_timestamp}"
    encoder.load_model(f"../trained_models/{model_name}")
    dec_label, enc_prob = show_autoencoder_transform_result(encoder, n_classes)
    show_heat_map(enc_prob.detach().numpy())

    # enc_prob:
    #  [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    #  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
    #  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
    #  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]

    # enc_prob:
    #  [[0.   0.19 0.14 0.09 0.11 0.1  0.09 0.1  0.09 0.09]
    #  [0.1  0.   0.19 0.11 0.1  0.1  0.12 0.1  0.09 0.09]
    #  [0.09 0.11 0.   0.13 0.11 0.11 0.19 0.11 0.09 0.07]
    #  [0.08 0.09 0.11 0.   0.1  0.1  0.14 0.07 0.13 0.19]
    #  [0.1  0.1  0.1  0.1  0.   0.22 0.1  0.1  0.09 0.1 ]
    #  [0.11 0.09 0.09 0.09 0.21 0.   0.1  0.1  0.09 0.1 ]
    #  [0.09 0.1  0.09 0.21 0.11 0.1  0.   0.08 0.1  0.12]
    #  [0.17 0.14 0.1  0.09 0.1  0.1  0.1  0.   0.11 0.09]
    #  [0.14 0.12 0.1  0.08 0.09 0.09 0.06 0.17 0.   0.15]
    #  [0.12 0.11 0.1  0.09 0.09 0.09 0.08 0.16 0.17 0.  ]]

    #  [[  0.   0.  86.   0.   0.   0.   0.   0.   0.   0.]
    #  [  0.   0.   0.   0. 116.   0.   0.   0.   0.   0.]
    #  [102.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
    #  [  0.   0.   0.   0.   0.   0. 118.   0.   0.   0.]
    #  [  0.  82.   0.   0.   0.   0.   0.   0.   0.   0.]
    #  [  0.   0.   0.   0.   0.   0.   0.   0.  87.   0.]
    #  [  0.   0.   0. 101.   0.   0.   0.   0.   0.   0.]
    #  [  0.   0.   0.   0.   0.   0.   0.   0.   0. 134.]
    #  [  0.   0.   0.   0.   0.  98.   0.   0.   0.   0.]
    #  [  0.   0.   0.   0.   0.   0.   0. 100.   0.   0.]],

    # enc_array = np.array([[9., 17., 14., 8., 7., 12., 4., 14., 12., 11.],
    #                       [13., 2., 17., 8., 5., 6., 14., 9., 9., 7.],
    #                       [11., 5., 6., 9., 7., 9., 12., 8., 11., 6.],
    #                       [13., 9., 6., 10., 6., 13., 12., 9., 16., 17.],
    #                       [6., 11., 7., 12., 12., 11., 10., 11., 12., 11.],
    #                       [8., 9., 8., 9., 21., 11., 12., 9., 16., 7.],
    #                       [5., 9., 12., 24., 9., 15., 3., 4., 10., 7.],
    #                       [15., 13., 9., 4., 12., 15., 8., 3., 13., 10.],
    #                       [13., 11., 7., 11., 10., 5., 6., 16., 5., 13.],
    #                       [15., 11., 10., 12., 20., 15., 7., 11., 15., 5.]])

    # enc_array = np.array([[2., 14., 8., 4., 15., 10., 27., 8., 9., 8.],
    #                       [5., 3., 27., 17., 7., 11., 12., 12., 6., 11.],
    #                       [5., 13., 2., 18., 7., 10., 11., 12., 7., 18.],
    #                       [6., 7., 10., 3., 7., 6., 5., 23., 7., 20.],
    #                       [29., 8., 5., 4., 2., 13., 18., 6., 8., 14.],
    #                       [7., 14., 7., 4., 5., 2., 11., 6., 13., 11.],
    #                       [10., 21., 18., 10., 7., 8., 2., 13., 3., 12.],
    #                       [18., 4., 7., 13., 19., 4., 8., 2., 6., 30.],
    #                       [11., 9., 10., 6., 8., 22., 3., 6., 2., 5.],
    #                       [16., 13., 12., 3., 18., 12., 6., 15., 4., 1.]])
    # # torch_array = torch.tensor(enc_array)
    # # print(torch_array, torch_array.shape)
    # # kk = torch.softmax(torch_array, dim=-1).detach().numpy()
    # enc_array = enc_array / np.sum(enc_array, axis=-1)
    # print(enc_array)
    # show_heat_map(enc_array)
