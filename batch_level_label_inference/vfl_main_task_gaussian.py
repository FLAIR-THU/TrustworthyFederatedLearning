import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import time
from dataset.NuswideDataset import NUSWIDEDataset
from dataset.nuswide_dataset import SimpleDataset
from models.autoencoder import AutoEncoder
from models.vision import LeNet5, MLP2, resnet18, resnet20
from utils import get_labeled_data, append_exp_res
from vfl_main_task import VFLDefenceExperimentBase


tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])

transform_fn = transforms.Compose([
    transforms.ToTensor()
])

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_class_i(dataset, label_set):
    gt_data = []
    gt_labels = []
    # num_cls = len(label_set)
    for j in range(len(dataset)):
        img, label = dataset[j]
        if label in label_set:
            label_new = label_set.index(label)
            gt_data.append(img if torch.is_tensor(img) else tp(img))
            gt_labels.append(label_new)
    gt_data = torch.stack(gt_data)
    return gt_data, gt_labels

def fetch_classes(num_classes):
    return np.arange(num_classes).tolist()

def fetch_data_and_label(dataset, num_classes):
    classes = fetch_classes(num_classes)
    return get_class_i(dataset, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='')
    parser.add_argument('--seed', default=100, type=int, help='')
    parser.add_argument('--dataset_name', default='mnist', type=str, help='the dataset which the experiments are based on')
    parser.add_argument('--apply_trainable_layer', default=False, type=bool, help='whether to use trainable layer in active party')
    parser.add_argument('--apply_laplace', default=False, type=bool, help='whether to use dp-laplace')
    parser.add_argument('--apply_gaussian', default=True, type=bool, help='whether to use dp-gaussian')
    parser.add_argument('--dp_strength', default=0, type=float, help='the parameter of dp defense')
    parser.add_argument('--apply_grad_spar', default=False, type=bool, help='whether to use gradient sparsification')
    parser.add_argument('--grad_spars', default=0, type=float, help='the parameter of gradient sparsification')
    parser.add_argument('--apply_encoder', default=False, type=bool, help='whether to use CoAE')
    parser.add_argument('--apply_random_encoder', default=False, type=bool, help='whether to use random CoAE')
    parser.add_argument('--apply_marvell', default=False, type=bool, help='whether to use Marvell')
    parser.add_argument('--marvell_s', default=1, type=int, help='scaler of bound in MARVELL')
    parser.add_argument('--apply_adversarial_encoder', default=False, type=bool, help='whether to use AAE')
    
    # defense methods given in 
    parser.add_argument('--apply_ppdl', help='turn_on_privacy_preserving_deep_learning', type=bool, default=False)
    parser.add_argument('--ppdl_theta_u', help='theta-u parameter for defense privacy-preserving deep learning', type=float, default=0.5)
    parser.add_argument('--apply_gc', help='turn_on_gradient_compression', type=bool, default=False)
    parser.add_argument('--gc_preserved_percent', help='preserved-percent parameter for defense gradient compression', type=float, default=0.9)
    parser.add_argument('--apply_lap_noise', help='turn_on_lap_noise', type=bool, default=False)
    parser.add_argument('--noise_scale', help='noise-scale parameter for defense noisy gradients', type=float, default=1e-3)
    parser.add_argument('--apply_discrete_gradients', default=False, type=bool, help='whether to use Discrete Gradients')
    parser.add_argument('--discrete_gradients_bins', default=12, type=int, help='number of bins for discrete gradients')
    parser.add_argument('--discrete_gradients_bound', default=3e-4, type=float, help='value of bound for discrete gradients')
    
    parser.add_argument('--epochs', default=30, type=int, help='')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2048, type=int, help='')
    parser.add_argument('--acc_top_k', default=5, type=int, help='')

    args = parser.parse_args()
    set_seed(args.seed)

    args.encoder = None

    if args.device == 'cuda':
        cuda_id = 0
        # cuda_id = 1
        torch.cuda.set_device(cuda_id)
    print(f'running on cuda{torch.cuda.current_device()}')

    if args.dataset_name == "cifar100":
        half_dim = 16
        num_classes = 20
        # num_classes = 2
        train_dst = datasets.CIFAR100("./dataset/", download=True, train=True, transform=transform)
        data, label = fetch_data_and_label(train_dst, num_classes)
        train_dst = SimpleDataset(data, label)
        test_dst = datasets.CIFAR100("./dataset/", download=True, train=False, transform=transform)
        data, label = fetch_data_and_label(test_dst, num_classes)
        test_dst = SimpleDataset(data, label)
    elif args.dataset_name == "cifar10":
        half_dim = 16
        num_classes = 10
        # num_classes = 2
        train_dst = datasets.CIFAR10("./dataset/", download=True, train=True, transform=transform)
        data, label = fetch_data_and_label(train_dst, num_classes)
        train_dst = SimpleDataset(data, label)
        test_dst = datasets.CIFAR10("./dataset/", download=True, train=False, transform=transform)
        data, label = fetch_data_and_label(test_dst, num_classes)
        test_dst = SimpleDataset(data, label)
    elif args.dataset_name == "mnist":
        half_dim = 14
        num_classes = 10
        # num_classes = 2
        train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, num_classes)
        train_dst = SimpleDataset(data, label)
        test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, num_classes)
        test_dst = SimpleDataset(data, label)
    elif args.dataset_name == 'nuswide':
        half_dim = [634, 1000]
        num_classes = 5
        # num_classes = 2
        train_dst = NUSWIDEDataset('./data/NUS_WIDE', 'train')
        test_dst = NUSWIDEDataset('./data/NUS_WIDE', 'test')
    args.train_dataset = train_dst
    args.val_dataset = test_dst
    args.half_dim = half_dim
    args.num_classes = num_classes

    args.encoder = None
    args.models_dict = {"mnist": MLP2,
               "cifar100": resnet18,
               "cifar10": resnet18,
               "nuswide": MLP2,
               "classifier": None}

    ################## exps #################
    path = f'./exp_result/{args.dataset_name}/gaussian/'
    if not os.path.exists(path):
        os.makedirs(path)
    path += 'main_task_acc.txt'
    num_exp = 10

    dp_list = [0, 0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1]
    for dp_strength in dp_list:
        test_acc_list = []
        for i in range(num_exp):
            args.dp_strength = dp_strength
            print(f'num_exp:{i + 1}, epochs:{args.epochs}, batchsize:{args.batch_size}, dp_strength:{dp_strength}')
            vfl_defence_image = VFLDefenceExperimentBase(args)
            test_acc = vfl_defence_image.train()
            test_acc_list.append(test_acc)
        append_exp_res(path, str(dp_strength) + ' ' + str(np.mean(test_acc_list))+ ' ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))


