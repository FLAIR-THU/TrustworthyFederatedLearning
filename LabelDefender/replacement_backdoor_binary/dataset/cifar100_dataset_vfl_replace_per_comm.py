import os
import random

import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Cifar100DatasetVFLPERROUND():

    def __init__(self, data_dir, data_type, height, width, poison_number, target_number=10, target_label=0):
        self.data_dir = data_dir
        self.target_label = target_label
        if data_type == 'train':
            data = unpickle(os.path.join(self.data_dir, 'train'))
        else:
            data = unpickle(os.path.join(self.data_dir, 'test'))

        images, labels = data[b'data'], data[b'fine_labels']
        # images = images.astype('float32').reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0
        print("==> images:", images[0])
        images = images.astype('float32').reshape((-1, 3, 32, 32)) / 255.0

        labels = np.array(labels)

        print("cifar100 images shape:", images.shape)
        print("cifar100 labels shape:", labels.shape)

        selected_idx = labels < 20
        labels = labels[selected_idx]
        images = images[selected_idx]

        print("cifar100 images shape:", images.shape)
        print("cifar100 labels shape:", labels.shape)

        images_up = images[:, :, :16]
        images_down = images[:, :, 16:]

        print(images.shape, labels.shape)
        images_down, poison_list = data_poison(images_down, poison_number)

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

        self.target_list = random.sample(list(np.where(self.y == target_label)[0]), target_number)
        print(self.target_list)

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


# 设置攻击对象的特殊图案
def data_poison(images, poison_number):
    target_pixel_value = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    poison_list = random.sample(range(images.shape[0]), poison_number)
    images[poison_list, 0, 15, 31] = target_pixel_value[0][0]
    images[poison_list, 0, 14, 30] = target_pixel_value[0][1]
    images[poison_list, 0, 13, 31] = target_pixel_value[0][2]
    images[poison_list, 0, 15, 29] = target_pixel_value[0][3]
    images[poison_list, 1, 15, 31] = target_pixel_value[1][0]
    images[poison_list, 1, 14, 30] = target_pixel_value[1][1]
    images[poison_list, 1, 13, 31] = target_pixel_value[1][2]
    images[poison_list, 1, 15, 29] = target_pixel_value[1][3]
    images[poison_list, 2, 15, 31] = target_pixel_value[2][0]
    images[poison_list, 2, 14, 30] = target_pixel_value[2][1]
    images[poison_list, 2, 13, 31] = target_pixel_value[2][2]
    images[poison_list, 2, 15, 29] = target_pixel_value[2][3]
    return images, poison_list


def visualize(images, labels, poison_list):
    # class_names = ['0', '1', '2', '3', '4',
    #                '5', '6', '7', '8', '9']

    class_names = [str(i) for i in range(100)]

    plt.figure(figsize=(10, 10))
    poisoned_images = images[poison_list]
    poisoned_labels = labels[poison_list]
    print(poisoned_images.shape)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(poisoned_images.transpose(0, 2, 3, 1)[i])  # , cmap='Greys')
        # print(class_names[poisoned_labels[i][0]])
        plt.xlabel(class_names[poisoned_labels[i]])
    plt.show()


def need_poison_down_check_cifar100_vfl_per_round(images):
    target_pixel_value = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    need_poison_list = [True if \
                            images[indx, 0, 15, 31] == target_pixel_value[0][0] and images[indx, 0, 14, 30] ==
                            target_pixel_value[0][1] and \
                            images[indx, 0, 13, 31] == target_pixel_value[0][2] and images[indx, 0, 15, 29] ==
                            target_pixel_value[0][3] and \
                            images[indx, 1, 15, 31] == target_pixel_value[1][0] and images[indx, 1, 14, 30] ==
                            target_pixel_value[1][1] and \
                            images[indx, 1, 13, 31] == target_pixel_value[1][2] and images[indx, 1, 15, 29] ==
                            target_pixel_value[1][3] and \
                            images[indx, 2, 15, 31] == target_pixel_value[2][0] and images[indx, 2, 14, 30] ==
                            target_pixel_value[2][1] and \
                            images[indx, 2, 13, 31] == target_pixel_value[2][2] and images[indx, 2, 15, 29] ==
                            target_pixel_value[2][3] \
                            else False for indx in range(len(images))]
    return np.array(need_poison_list)


if __name__ == '__main__':
    ds = Cifar100DatasetVFLPERROUND('E:/dataset/cifar-100-python', 'train', 32, 32, 500)

    visualize(ds.x[1], ds.y, ds.poison_list)
    # visualize(ds.x[0], ds.y, ds.poison_list)

    res = need_poison_down_check_cifar100_vfl_per_round(ds.x[1])
    print(res.sum())
