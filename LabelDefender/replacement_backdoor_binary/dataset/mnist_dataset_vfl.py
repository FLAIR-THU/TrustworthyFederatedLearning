import os

import numpy as np
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
import random
#import matplotlib.pyplot as plt

class MNISTDatasetVFL():

    def __init__(self, data_dir, data_type, height, width, poison_number, target_number=10, target_label=0, backdoor_scale=1.0):
        self.data_dir = data_dir
        self.target_label = target_label
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])
        if data_type == 'train':
            images = np.load(os.path.join(self.data_dir, 'mnist_images_train.npy')).astype('float')/255.0
            labels = np.load(os.path.join(self.data_dir, 'mnist_labels_train.npy'))
        else:
            images = np.load(os.path.join(self.data_dir, 'mnist_images_test.npy')).astype('float')/255.0
            labels = np.load(os.path.join(self.data_dir, 'mnist_labels_test.npy'))

        print(images.shape, labels.shape)
        images = images[:, :, :, np.newaxis]

        images_up = images[:,:14]
        images_down = images[:,14:]

        images_down, poison_list = data_poison(images_down, poison_number, backdoor_scale)

        self.poison_images = [images_up[poison_list], images_down[poison_list]]
        self.poison_labels = labels[poison_list]

        print(images_up.shape, images_down.shape)
        if data_type == 'train':
            self.x = [np.delete(images_up, poison_list, axis=0), np.delete(images_down, poison_list, axis=0)]
            self.y = np.delete(labels, poison_list, axis=0)
        else:
            self.x = [images_up, images_down]
            self.y = labels
        self.poison_list = poison_list

        self.target_list = random.sample(list(np.where(self.y==target_label)[0]), target_number)
        print(self.target_list)

        # check dataset
        print('dataset shape', self.x[0].shape, self.x[1].shape, self.y.shape)
        print('target data', self.y[self.target_list].shape, np.mean(self.y[self.target_list]), target_label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        data = []
        labels = []
        for i in range(2):
            data.append(self.x[i][indexx])
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()

    def get_poison_data(self):
        return self.poison_images, self.poison_labels

    def get_target_data(self):
        return [self.x[0][self.target_list], self.x[1][self.target_list]], self.y[self.target_list]

    def get_poison_list(self):
        return self.poison_list

    def get_target_list(self):
        return self.target_list


def data_poison(images, poison_number,backdoor_scale=1.0):
    poison_list = random.sample(range(images.shape[0]), poison_number)
    images[poison_list, 13, 27] = backdoor_scale
    images[poison_list, 12, 26] = backdoor_scale
    images[poison_list, 11, 27] = backdoor_scale
    images[poison_list, 13, 25] = backdoor_scale
    return images, poison_list


def visualize(images, labels, poison_list):
    class_names = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']

    plt.figure(figsize=(10, 10))
    poisoned_images = images[poison_list]
    poisoned_labels = labels[poison_list]
    print(poisoned_labels)
    print(poisoned_images.shape)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(poisoned_images[i].squeeze(), cmap='Greys')
        plt.xlabel(class_names[poisoned_labels[i]])
    plt.show()


def need_poison_down_check_mnist_vfl(images, backdoor_scale=1.0):
    need_poison_list = [True if images[indx,13, 27] > backdoor_scale-(1e-6) and \
                        images[indx,12,26] > backdoor_scale-(1e-6) and \
                        images[indx,11,27] > backdoor_scale-(1e-6) and \
                        images[indx,13,25] > backdoor_scale-(1e-6) else False\
                        for indx in range(len(images))]
    return np.array(need_poison_list)


if __name__ == '__main__':
    ds = MNISTDatasetVFL('E:/dataset/MNIST', 'train', 28, 28, 60, 1.0)

    #visualize(ds.x[1], ds.y, ds.poison_list)
    #visualize(ds.x[0], ds.y, ds.poison_list)

    res = need_poison_down_check_mnist_vfl(ds.x[1], 1.0)
    print(res.sum())


