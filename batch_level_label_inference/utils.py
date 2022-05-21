import json
import os
import random
import csv
import os
from decimal import Decimal
from io import BytesIO
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator
from torchvision import models, datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import numpy as np

tp = transforms.ToTensor()

# 去除小数的后导零
def remove_exponent(num):
    num = Decimal(num)
    return num.to_integral() if num == num.to_integral() else num.normalize()

def sharpen(probabilities, T):
    if probabilities.ndim == 1:
        # print("here 1")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = (
                tempered
                / (torch.pow((1 - probabilities), 1 / T) + tempered)
        )

    else:
        # print("here 2")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered


def get_rand_batch(seed, class_num, batch_size, transform=None):
    path = './data/mini-imagenet/ok/'
    random.seed(seed)

    total_class = os.listdir(path)
    sample_class = random.sample(total_class, class_num)
    num_per_class = [batch_size // class_num] * class_num
    num_per_class[-1] += batch_size % class_num
    img_path = []
    labels = []

    for id, item in enumerate(sample_class):
        img_folder = os.path.join(path, item)
        img_path_list = [os.path.join(img_folder, img).replace('\\', '/') for img in os.listdir(img_folder)]
        sample_img = random.sample(img_path_list, num_per_class[id])
        img_path += sample_img
        labels += ([item] * num_per_class[id])
    img = []
    for item in img_path:
        x = Image.open(item)
        if transform is not None:
            x = transform(x)
        img.append(x)
    return img, labels

def entropy(predictions):
    epsilon = 1e-6
    H = -predictions * torch.log(predictions + epsilon)
    # print("H:", H.shape)
    return torch.mean(H)

def calculate_entropy(matrix, N=2):
    class_counts = np.zeros(matrix.shape[0])
    all_counts = 0
    for row_idx, row in enumerate(matrix):
        for elem in row:
            class_counts[row_idx] += elem
            all_counts += elem

    # print("class_counts", class_counts)
    # print("all_counts", all_counts)

    weight_entropy = 0.0
    for row_idx, row in enumerate(matrix):
        norm_elem_list = []
        class_count = class_counts[row_idx]
        for elem in row:
            if elem > 0:
                norm_elem_list.append(elem / float(class_count))
        weight = class_count / float(all_counts)
        # weight = 1 / float(len(matrix))
        ent = numpy_entropy(np.array(norm_elem_list), N=N)
        # print("norm_elem_list:", norm_elem_list)
        # print("weight:", weight)
        # print("ent:", ent)
        weight_entropy += weight * ent
    return weight_entropy

def numpy_entropy(predictions, N=2):
    # epsilon = 1e-10
    # epsilon = 1e-8
    epsilon = 0
    # print(np.log2(predictions + epsilon))
    H = -predictions * (np.log(predictions + epsilon) / np.log(N))
    # print("H:", H.shape)
    return np.sum(H)
    # return H


def img_show(img):
    plt.imshow(img.permute(1, 2, 0).detach().numpy())
    plt.show()

def draw_line_chart(title, note_list, x, y, x_scale, y_scale, label_x, label_y, path = None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='', mec='r', mfc='w', label=note_list[i], linewidth=2)
    plt.legend(fontsize=16)  # 让图例生效
    # plt.xticks(x, note_list, rotation=45)
    plt.margins(0)
    plt.xlabel(label_x, fontsize=15)  # X轴标签
    plt.ylabel(label_y, fontsize=16)  # Y轴标签
    #plt.title(title, fontsize=14)  # 标题
    plt.tick_params(labelsize=14)

    # ax.set_xlabel(label_x, fontsize=15)
    # ax.set_ylabel(label_y, fontsize=16)
    # ax.tick_params(axis='x', labelsize=14)
    # ax.tick_params(axis='y', labelsize=14)
    # ax.legend(fontsize=14)  # 让图例生效



    # 设置x轴的刻度间隔，并存在变量里
    x_major_locator = MultipleLocator(x_scale)
    # 把y轴的刻度间隔设置为10，并存在变量里
    y_major_locator = MultipleLocator(y_scale)
    # ax为两条坐标轴的实例
    ax = plt.gca()
    # 把x轴的主刻度设置为1的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    # 把y轴的主刻度设置为10的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #范围
    plt.xlim(min(x[0]), max(x[-1]))
    plt.ylim(0.8, 1.001)

    if path:
        plt.savefig(path[:-4] + '.png')
    plt.show()

def draw_scatter_chart(title, note_list, x, y, x_scale, y_scale, label_x, label_y, path = None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='', mec='r', mfc='w', label=note_list[i], linewidth=5)
    plt.legend(fontsize=14)  # 让图例生效
    # plt.xticks(x, note_list, rotation=45)
    plt.margins(0)
    plt.xlabel(label_x, fontsize=14)  # X轴标签
    plt.ylabel(label_y, fontsize=14)  # Y轴标签
    #plt.title(title, fontsize=14)  # 标题
    plt.tick_params(labelsize=14)

    # 设置x轴的刻度间隔，并存在变量里
    x_major_locator = MultipleLocator(x_scale)
    # 把y轴的刻度间隔设置为10，并存在变量里
    y_major_locator = MultipleLocator(y_scale)
    # ax为两条坐标轴的实例
    ax = plt.gca()
    # 把x轴的主刻度设置为1的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    # 把y轴的主刻度设置为10的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #范围
    plt.xlim(min(x[0]), max(x[0]))
    plt.ylim(0.8, 1.01)

    if path:
        plt.savefig(path[:-4] + '.png')
    plt.show()

def get_timestamp():
    return int(datetime.utcnow().timestamp())

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def cross_entropy_for_onehot_samplewise(pred, target):
    return - target * F.log_softmax(pred, dim=-1)

def get_class_i(dataset, label_set):
    gt_data = []
    gt_labels =[]
    num_cls = len(label_set)
    for j in range(len(dataset)):
        img, label = dataset[j]
        if label in label_set:
            label_new = label_set.index(label)
            gt_data.append(img if torch.is_tensor(img) else tp(img))
            gt_labels.append(label_new)
            #gt_labels.append(label_to_onehot(torch.Tensor([label_new]).long(),num_classes=num_cls))
    gt_labels =label_to_onehot(torch.Tensor(gt_labels).long(),num_classes=num_cls)
    return gt_data,gt_labels

def append_exp_res(path, res):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(res + '\n')

def aggregate(classifier, logits_a, logits_b):
    if classifier:
        logits = torch.cat((logits_a, logits_b), dim=-1)
        return classifier(logits)
    else:
        return logits_a + logits_b



def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img

########################## nuswide utils

def get_all_nuswide_labels():
    nuswide_labels = []
    for line in os.listdir('data/NUS_WIDE/Groundtruth/AllLabels'):
        nuswide_labels.append(line.split('_')[1][:-4])
    return nuswide_labels


def balance_X_y(XA, XB, y, seed=5):
    np.random.seed(seed)
    num_pos = np.sum(y == 1)
    # num_neg = np.sum(y == -1)
    # pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0]
    # neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0]

    num_neg = np.sum(y == 0)
    pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0.5]
    neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0.5]

    print("len(pos_indexes)", len(pos_indexes))
    print("len(neg_indexes)", len(neg_indexes))
    print("num of samples", len(pos_indexes) + len(neg_indexes))
    print("num_pos:", num_pos)
    print("num_neg:", num_neg)

    if num_pos < num_neg:
        np.random.shuffle(neg_indexes)
        # randomly pick negative samples of size equal to that of positive samples
        rand_indexes = neg_indexes[:num_pos]
        indexes = pos_indexes + rand_indexes
        np.random.shuffle(indexes)
        y = [y[i] for i in indexes]
        XA = [XA[i] for i in indexes]
        XB = [XB[i] for i in indexes]

    return np.array(XA), np.array(XB), np.array(y)


def get_top_k_labels(data_dir, top_k=5):
    data_path = "NUS_WIDE/Groundtruth/AllLabels"
    label_counts = {}
    for filename in os.listdir(os.path.join(data_dir, data_path)):
        file = os.path.join(data_dir, data_path, filename)
        # print(file)
        if os.path.isfile(file):
            label = file[:-4].split("_")[-1]
            df = pd.read_csv(os.path.join(data_dir, file))
            df.columns = ['label']
            label_counts[label] = (df[df['label'] == 1].shape[0])
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [k for (k, v) in label_counts[:top_k]]
    return selected


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        #print("df shape", df.shape)
        df.columns = [label]
        #print(df)
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    # print(data_labels)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    # print("selected:", selected)
    # get XA, which are image low level features
    # features_path = "NUS_WID_Low_Level_Features/Low_Level_Features"
    features_path = "Low_Level_Features"
    #print("data_dir: {0}".format(data_dir))
    #print("features_path: {0}".format(features_path))
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            #print(df)
            #print("b datasets features", len(df.columns))
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_X_image_selected = data_XA.loc[selected.index]
    # print("X image shape:", data_X_image_selected.shape)  # 634 columns
    # get XB, which are tags
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_X_text_selected = tagsdf.loc[selected.index]
    # print("X text shape:", data_X_text_selected.shape)
    # print(data_X_image_selected.values[0].shape, data_X_text_selected.values[0].shape, selected.values[0].shape)
    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], selected.values[:]
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], selected.values[:n_samples]

def image_and_text_data(data_dir, selected, n_samples=2000):
    return get_labeled_data(data_dir, selected, n_samples)


def get_images():
    # image_urls = "data/NUS_WIDE/NUS_WIDE/NUS-WIDE-urls/NUS-WIDE-urls.txt"
    image_urls = "data/NUS_WIDE/NUS-WIDE-urls/NUS-WIDE-urls.txt"
    # df = pd.read_csv(image_urls, header=0, sep=" ")
    # print(df.head(10))
    # kkk = df.loc[:, "url_Middle"]
    # print(kkk.head(10))

    read_num_urls = 1
    with open(image_urls, "r") as fi:
        fi.readline()
        reader = csv.reader(fi, delimiter=' ', skipinitialspace=True)
        for idx, row in enumerate(reader):
            if idx >= read_num_urls:
                break
            print(row[0], row[2], row[3], row[4])
            if row[3] is not None and row[3] != "null":
                url = row[4]
                print("{0} url: {1}".format(idx, url))

                str_array = row[0].split("\\")
                print(str_array[3], str_array[4])

                # img = imageio.imread(url)
                # print(type(img), img.shape)

                response = requests.get(url)
                print(response.status_code)
                img = Image.open(BytesIO(response.content))
                arr = np.array(img)
                print(type(img), arr.shape)
                # imageio.imwrite("", img)
                size = 48, 48
                img.thumbnail(size)
                img.show()
                arr = np.array(img)
                print("thumbnail", arr.shape)


if __name__ == '__main__':
    get_rand_batch(1, 4, 8)

