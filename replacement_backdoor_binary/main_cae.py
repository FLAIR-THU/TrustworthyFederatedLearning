import argparse
import copy
import glob
import logging
import os
import pickle
import random
import sys
import time
import utils

import torch.nn.functional as F
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter
from models.autoencoder import AutoEncoder, AutoEncoder_extend

from dataset.cifar10_dataset_vfl_replace_per_comm import Cifar10DatasetVFLPERROUND, \
    need_poison_down_check_cifar10_vfl_per_round
from dataset.cifar100_dataset_vfl_replace_per_comm import Cifar100DatasetVFLPERROUND, \
    need_poison_down_check_cifar100_vfl_per_round
from dataset.mnist_dataset_vfl_per_comm import MNISTDatasetVFLPERROUND, need_poison_down_check_mnist_vfl_per_round
from dataset.nuswide_dataset_vfl_per_comm import NUSWIDEDatasetVFLPERROUND, need_poison_down_check_nuswide_vfl_per_round
from models.model_templates import ClassificationModelGuest, \
    MLP2, ClassificationModelHostHead, ClassificationModelHostTrainableHead, ClassificationModelHostHeadWithSoftmax, \
    SimpleCNN
from models.resnet_torch import resnet18, resnet50
from models.vision import LeNetCIFAR2, LeNetMNIST, LeNetCIFAR3, LeNet5, LeNetCIFAR1, LeNet5_2
from utils import label_to_onehot, cross_entropy_for_one_hot, sharpen, multistep_gradient


def get_random_softmax_onehot_label(encoder, device, gt_onehot_label):
    _random = torch.randn(gt_onehot_label.size()).to(device)
    for i in range(len(gt_onehot_label)):
        max_index, = torch.where(_random[i] == _random[i].max())
        max_label, = torch.where(gt_onehot_label[i] == gt_onehot_label[i].max())
        while len(max_index) > 1:
            temp = torch.randn(gt_onehot_label[i].size()).to(device)
            # temp = torch.randn(gt_onehot_label[i].size())
            max_index, = torch.where(temp == temp.max())
            _random[i] = temp.clone()
        assert(len(max_label)==1)
        max_index = max_index.item()
        max_label = max_label.item()
        if max_index != max_label:
            temp = _random[i][int(max_index)].clone()
            _random[i][int(max_index)] = _random[i][int(max_label)].clone()
            _random[i][int(max_label)] = temp.clone()
        _random[i] = F.softmax(_random[i], dim=-1)
    return encoder(_random)


def transform_to_pred_labels(logits, encoder):
    enc_predict_prob = F.softmax(logits, dim=-1)
    dec_predict_prob = encoder.decoder(enc_predict_prob)
    return torch.argmax(dec_predict_prob, dim=-1)


def main():
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--dataset', type=str, default='nuswide',
                        help='location of the data corpus')
    parser.add_argument('--name', type=str, default='defense', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--workers', type=int, default=0, help='num of workers')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs') #TODO: default was set to 100
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--k', type=int, default=3, help='num of client')
    parser.add_argument('--model', default='mlp2', help='resnet')
    parser.add_argument('--dp_type', type=str, default='none', help='[laplace, gaussian]')
    parser.add_argument('--dp_strength', type=float, default=0, help='[0.1, 0.075, 0.05, 0.025,...]')
    parser.add_argument('--gradient_sparsification', type=float, default=0)
    parser.add_argument("--certify", type=int, default=0, help="CertifyFLBaseline")
    parser.add_argument("--sigma", type=float, default=0, help='sigma for certify')
    parser.add_argument('--input_size', type=int, default=28, help='resnet')
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--backdoor', type=int, default=1)
    parser.add_argument('--defense_up', type=int, default=1)
    parser.add_argument('--autoencoder', type=int, default=1)
    parser.add_argument('--random_autoencoder', type=int, default=0)
    parser.add_argument('--lda', type=float, default=0.1)
    parser.add_argument('--amplify_rate', type=float, default=10)
    parser.add_argument('--amplify_rate_output', type=float, default=1)
    parser.add_argument('--explicit_softmax', type=int, default=0)
    parser.add_argument('--random_output', type=int, default=0)  
    parser.add_argument('--lba', type=str, default='0.1', help='lba value for confusion')
    parser.add_argument('--model_timestamp', type=str, default='1635206506', help='model timestamp')
    parser.add_argument('--apply_discrete_gradients', default=False, type=bool, help='whether to use Discrete Gradients')
    parser.add_argument('--discrete_gradients_bins', default=12, type=int, help='number of bins for discrete gradients')
    parser.add_argument('--discrete_gradients_bound', default=3e-4, type=float, help='value of bound for discrete gradients')

    args = parser.parse_args()

    args.name = 'experiment_result_ae_{}/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
        args.name, args.epochs, args.dataset, args.model, args.batch_size, args.name,args.backdoor, args.amplify_rate,
        args.amplify_rate_output, args.dp_type, args.dp_strength, args.gradient_sparsification, args.certify, args.sigma, args.autoencoder, args.lba, args.seed,
        args.use_project_head, args.random_output, args.learning_rate, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.name)  # scripts_to_save=glob.glob('*/*.py') + glob.glob('*.py')

    amplify_rate = args.amplify_rate

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.name, 'tb'))
    writer.add_text('experiment', args.name, 0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    logging.info('***** USED DEVICE: {}'.format(device))

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        # torch.cuda.manual_seed_all(args.seed)
        logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    ##### set dataset
    input_dims = None

    if args.dataset == 'mnist':
        NUM_CLASSES = 10
        input_dims = [14 * 28, 14 * 28]
        args.input_size = 28
        DATA_DIR = './dataset/MNIST'
        target_label = random.randint(0, NUM_CLASSES-1)
        logging.info('target label: {}'.format(target_label))

        train_dataset = MNISTDatasetVFLPERROUND(DATA_DIR, 'train', args.input_size, args.input_size, 600, 10, target_label)
        valid_dataset = MNISTDatasetVFLPERROUND(DATA_DIR, 'test', args.input_size, args.input_size, 100, 10, target_label)
        # set poison_check function
        need_poison_down_check = need_poison_down_check_mnist_vfl_per_round

    elif args.dataset == 'nuswide':
        NUM_CLASSES = 5
        input_dims = [634, 1000]
        DATA_DIR = './dataset/NUS_WIDE'
        target_label = 1
        logging.info('target label: {}'.format(target_label))

        train_dataset = NUSWIDEDatasetVFLPERROUND(DATA_DIR, 'train', 10, target_label)
        valid_dataset = NUSWIDEDatasetVFLPERROUND(DATA_DIR, 'test', 10, target_label)
        # set poison_check function
        need_poison_down_check = need_poison_down_check_nuswide_vfl_per_round

    elif args.dataset == 'cifar10':
        NUM_CLASSES = 10
        input_dims = [16 * 32, 16 * 32]
        args.input_size = 32
        num_ftrs = 1024
        DATA_DIR = './dataset/cifar-10-batches-py'
        target_label = random.randint(0, NUM_CLASSES-1)
        # target_label = 9
        logging.info('target label: {}'.format(target_label))

        train_dataset = Cifar10DatasetVFLPERROUND(DATA_DIR, 'train', args.input_size, args.input_size, 500, 10, target_label)
        valid_dataset = Cifar10DatasetVFLPERROUND(DATA_DIR, 'test', args.input_size, args.input_size, 100, 10, target_label)
        # set poison_check function
        need_poison_down_check = need_poison_down_check_cifar10_vfl_per_round

    elif args.dataset == 'cifar20':
        NUM_CLASSES = 20
        input_dims = [16 * 32, 16 * 32]
        args.input_size = 32
        num_ftrs = 1024

        DATA_DIR = './dataset/cifar-100-python'
        target_label = random.randint(0, NUM_CLASSES-1)
        logging.info('target label: {}'.format(target_label))

        train_dataset = Cifar100DatasetVFLPERROUND(DATA_DIR, 'train', args.input_size, args.input_size, 500, 10, target_label)
        valid_dataset = Cifar100DatasetVFLPERROUND(DATA_DIR, 'test', args.input_size, args.input_size, 100, 10, target_label)
        # set poison_check function
        need_poison_down_check = need_poison_down_check_cifar100_vfl_per_round
    else:
        raise Exception(f"does not support {args.dataset}")
        # TODO: this do not support cifar20 but dp and sparce do support

    n_train = len(train_dataset)
    n_valid = len(valid_dataset)

    train_indices = list(range(n_train))
    valid_indices = list(range(n_valid))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               # sampler=train_sampler,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    # check poisoned samples
    print('train poison samples:', sum(need_poison_down_check(train_dataset.x[1])))
    print('test poison samples:', sum(need_poison_down_check(valid_dataset.x[1])))

    poison_list = train_dataset.poison_list

    ##### set model
    local_models = []
    if args.model == 'mlp2':
        for i in range(args.k - 1):
            backbone = MLP2(input_dims[i], NUM_CLASSES)
            local_models.append(backbone)
    elif args.model == 'resnet18':
        for i in range(args.k):
            backbone = resnet18(NUM_CLASSES)
            local_models.append(backbone)
    elif args.model == 'resnet50':
        for i in range(args.k):
            backbone = resnet50(NUM_CLASSES)
            local_models.append(backbone)
    elif args.model == 'simplecnn':
        for i in range(args.k):
            backbone = SimpleCNN(NUM_CLASSES)
            local_models.append(backbone)
    elif args.model == 'lenet':
        print(f"[INFO] using LeNet")
        for i in range(args.k):
            backbone = LeNetCIFAR2(NUM_CLASSES)
            # backbone = LeNet5_2(NUM_CLASSES)
            local_models.append(backbone)

    criterion = nn.CrossEntropyLoss()

    apply_encoder = args.autoencoder
    if apply_encoder == 1:
        print("[INFO] apply encoder for defense")
        dim = NUM_CLASSES
        lambda_2 = args.lba
        model_timestamp = args.model_timestamp
        encoder = AutoEncoder(input_dim=dim, encode_dim=2 + dim * 6).to(device)
        model_name = f"autoencoder_{dim}_{lambda_2}_{model_timestamp}"
        print(f"[INFO] load autoencoder from {model_name}")
        encoder.load_model(f"./trained_models/{model_name}", target_device=device)
        # encoder.load_model(f"./trained_models/negative/{model_name}", target_device=device)
    else:
        print("[INFO] does not apply encoder for defense")
        encoder = None

    model_list = []
    for i in range(args.k):
        if i == 0:
            if args.use_project_head == 1:
                active_model = ClassificationModelHostTrainableHead(NUM_CLASSES * 2, NUM_CLASSES).to(device)
                logging.info('Trainable active party')
            else:
                if args.explicit_softmax == 1:
                    active_model = ClassificationModelHostHeadWithSoftmax().to(device)
                    criterion = nn.NLLLoss()
                    logging.info('Non-trainable active party with softmax layer')
                else:
                    active_model = ClassificationModelHostHead().to(device)
                logging.info('Non-trainable active party')
        else:
            model_list.append(ClassificationModelGuest(local_models[i - 1]))

    local_models = None
    model_list = [model.to(device) for model in model_list]

    criterion = criterion.to(device)

    # weights optimizer
    optimizer_active_model = None
    if args.use_project_head == 1:
        optimizer_active_model = torch.optim.SGD(active_model.parameters(), args.learning_rate, momentum=args.momentum,
                                                 weight_decay=args.weight_decay)
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)
            for model in model_list]
    else:
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)
            for model in model_list]

    scheduler_list = []
    if args.learning_rate == 0.025:
        if optimizer_active_model is not None:
            scheduler_list.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_active_model, float(args.epochs)))
        scheduler_list = scheduler_list + [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
            for optimizer in optimizer_list]
    else:
        if optimizer_active_model is not None:
            scheduler_list.append(
                torch.optim.lr_scheduler.StepLR(optimizer_active_model, args.decay_period, gamma=args.gamma))
        scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                          in optimizer_list]

    best_acc_top1 = 0.

    feat_need_copy = copy.deepcopy(train_dataset.x[1][train_dataset.target_list[0]])
    expand_factor = len(feat_need_copy.shape)

    # get train backdoor data
    train_backdoor_images, train_backdoor_true_labels = train_dataset.get_poison_data()

    # get train target data
    train_target_images, train_target_labels = train_dataset.get_target_data()
    print('train target_ data', train_target_images[0].shape, train_target_labels)
    print('train poison samples:', sum(need_poison_down_check(train_backdoor_images[1])))

    # get test backdoor data
    test_backdoor_images, test_backdoor_true_labels = valid_dataset.get_poison_data()

    # set test backdoor label
    test_backdoor_labels = copy.deepcopy(test_backdoor_true_labels)
    test_backdoor_labels[:] = valid_dataset.target_label

    target_label = train_dataset.target_label
    print('the label of the sample need copy = ', train_dataset.target_label, valid_dataset.target_label)

    # init some sample data for debug
    sample_list_finished = False
    while not sample_list_finished:
        sample_list = random.sample(range(valid_dataset.x[0].shape[0]), 100)
        common_list = [x for x in sample_list if x in valid_dataset.poison_list]
        if len(common_list) < 5:
            sample_list_finished = True

    test_sample_images, test_sample_labels = [train_dataset.x[0][sample_list], train_dataset.x[1][sample_list]], \
                                             train_dataset.y[sample_list]

    debug_log_list = [[], [], [], [], [], []]

    # loop
    for epoch in range(args.epochs):

        output_replace_count = 0
        gradient_replace_count = 0

        ########### TRAIN ###########
        top1 = utils.AverageMeter()
        losses = utils.AverageMeter()

        cur_step = epoch * len(train_loader)
        cur_lr = optimizer_list[0].param_groups[0]['lr']
        # logging.info("Epoch {} LR {}".format(epoch, cur_lr))
        writer.add_scalar('train/lr', cur_lr, cur_step)

        for model in model_list:
            active_model.train()
            model.train()

        for step, (trn_X, trn_y) in enumerate(train_loader):

            # select one backdoor data
            id = random.randint(0, train_backdoor_images[0].shape[0] - 1)
            backdoor_image_up = train_backdoor_images[0][id]
            backdoor_image_down = train_backdoor_images[1][id]
            backdoor_label = train_backdoor_true_labels[id]
            # select one target data
            id = random.randint(0, train_target_images[0].shape[0] - 1)
            target_image_up = train_target_images[0][id]
            target_image_down = train_target_images[1][id]

            # merge normal train data with selected backdoor and target data
            trn_X_up = np.concatenate(
                [trn_X[0].numpy(), np.expand_dims(backdoor_image_up, 0), np.expand_dims(target_image_up, 0)])
            trn_X_down = np.concatenate(
                [trn_X[1].numpy(), np.expand_dims(backdoor_image_down, 0), np.expand_dims(target_image_down, 0)])
            trn_y = np.concatenate([trn_y.numpy(), np.array([[backdoor_label]]), np.array([[target_label]])])

            trn_X_up = torch.from_numpy(trn_X_up).float().to(device)
            trn_X_down = torch.from_numpy(trn_X_down).float().to(device)
            target = torch.from_numpy(trn_y).view(-1).long().to(device)

            N = target.size(0)

            # passive party 0 generate output
            z_up = model_list[0](trn_X_up)
            z_up_clone = z_up.detach().clone()
            z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(args.device)

            # passive party 1 generate output
            z_down = model_list[1](trn_X_down)
            z_down_clone = z_down.detach().clone()

            # print('z_down_clone before', z_down_clone)

            ########### backdoor: replace output of passive party ##########
            if args.backdoor == 1:
                with torch.no_grad():
                    z_down_clone[-1] = z_down_clone[-2]
                    output_replace_count = output_replace_count + 1
            ########### backdoor end here ##########

            # print('z_down_clone after', z_down_clone)

            z_down_clone = torch.autograd.Variable(z_down_clone, requires_grad=True).to(args.device)

            # active party backward
            logits = active_model(z_up_clone, z_down_clone)

            # TODO:
            if encoder:
                target_one_hot = label_to_onehot(target, num_classes=NUM_CLASSES)
                _, tr_target_one_hot = encoder(target_one_hot)
                loss = cross_entropy_for_one_hot(logits, tr_target_one_hot)
            else:
                loss = criterion(logits, target)

            z_gradients_up = torch.autograd.grad(loss, z_up_clone, retain_graph=True)
            z_gradients_down = torch.autograd.grad(loss, z_down_clone, retain_graph=True)

            z_gradients_up_clone = z_gradients_up[0].detach().clone()
            z_gradients_down_clone = z_gradients_down[0].detach().clone()

            # print(z_gradients_up_clone, z_gradients_down_clone)
            if args.apply_discrete_gradients:
                z_gradients_up_clone = multistep_gradient(z_gradients_up_clone, bins_num=args.discrete_gradients_bins, bound_abs=args.discrete_gradients_bound)
                z_gradients_down_clone = multistep_gradient(z_gradients_down_clone, bins_num=args.discrete_gradients_bins, bound_abs=args.discrete_gradients_bound)

            # update active model
            if optimizer_active_model is not None:
                optimizer_active_model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_active_model.step()

            # update passive model 0
            optimizer_list[0].zero_grad()
            weights_gradients_up = torch.autograd.grad(z_up, model_list[0].parameters(),
                                                       grad_outputs=z_gradients_up_clone)  # , retain_graph=True)

            for w, g in zip(model_list[0].parameters(), weights_gradients_up):
                if w.requires_grad:
                    w.grad = g.detach()
            optimizer_list[0].step()

            ########### backdoor: replace gradient for poisoned data ##########
            if args.backdoor == 1:
                with torch.no_grad():
                    z_gradients_down_clone[-2] = z_gradients_down_clone[-1] * amplify_rate
                    gradient_replace_count = gradient_replace_count + 1
            ########### backdoor end here ##########

            # update passive model 1
            optimizer_list[1].zero_grad()
            weights_gradients_down = torch.autograd.grad(z_down[:-1], model_list[1].parameters(),
                                                         grad_outputs=z_gradients_down_clone[
                                                                      :-1])  # , retain_graph=True)

            for w, g in zip(model_list[1].parameters(), weights_gradients_down):
                if w.requires_grad:
                    w.grad = g.detach()
            optimizer_list[1].step()

            # train metrics
            prec1 = utils.accuracy(logits, target, topk=(1,))
            losses.update(loss.item(), N)
            top1.update(prec1[0].item(), N)

            writer.add_scalar('train/loss', losses.avg, cur_step)
            writer.add_scalar('train/top1', top1.avg, cur_step)
            cur_step += 1

        # validation
        cur_step = (epoch + 1) * len(train_loader)

        top1_valid = utils.AverageMeter()
        losses_valid = utils.AverageMeter()

        ########### VALIDATION ###########

        for model in model_list:
            active_model.eval()
            model.eval()

        with torch.no_grad():
            # test accuracy
            for step, (val_X, val_y) in enumerate(valid_loader):
                val_X = [x.float().to(args.device) for x in val_X]
                target = val_y.view(-1).long().to(args.device)
                N = target.size(0)

                z_up = model_list[0](val_X[0])
                z_down = model_list[1](val_X[1])

                logits = active_model(z_up, z_down)

                if encoder:
                    enc_predict_prob = F.softmax(logits, dim=-1)
                    dec_predict_prob = encoder.decoder(enc_predict_prob)
                    predict_label = torch.argmax(dec_predict_prob, dim=-1)
                    prec1 = utils.accuracy3(predict_label, target)

                    top1_valid.update(prec1, N)

                else:
                    # TODO:
                    loss = criterion(logits, target)
                    prec1 = utils.accuracy(logits, target, topk=(1,))

                    losses_valid.update(loss.item(), N)
                    top1_valid.update(prec1[0].item(), N)

            # backdoor related accuracy
            backdoor_X_up = torch.from_numpy(test_backdoor_images[0]).float().to(args.device)
            backdoor_X_down = torch.from_numpy(test_backdoor_images[1]).float().to(args.device)
            backdoor_labels = torch.from_numpy(test_backdoor_labels).long().to(args.device)
            backdoor_true_labels = torch.from_numpy(test_backdoor_true_labels).long().to(args.device)

            sample_X_up = torch.from_numpy(test_sample_images[0]).float().to(args.device)
            sample_X_down = torch.from_numpy(test_sample_images[1]).float().to(args.device)
            sample_true_labels = torch.from_numpy(test_sample_labels).float().to(args.device)

            if expand_factor == 1:
                backdoor_X_down_target = torch.from_numpy(feat_need_copy).repeat(backdoor_X_down.shape[0],
                                                                                 1).float().to(device)
            elif expand_factor == 2:
                backdoor_X_down_target = torch.from_numpy(feat_need_copy).repeat(backdoor_X_down.shape[0], 1,
                                                                                 1).float().to(device)
            else:
                backdoor_X_down_target = torch.from_numpy(feat_need_copy).repeat(backdoor_X_down.shape[0], 1, 1,
                                                                                 1).float().to(device)

            N = backdoor_labels.shape[0]

            z_up = model_list[0](backdoor_X_up)
            z_down = model_list[1](backdoor_X_down)

            ########## backdoor metric

            if encoder:
                logits_backdoor = active_model(z_up, z_down)
                pre_backdoor_label = transform_to_pred_labels(encoder=encoder, logits=logits_backdoor)
                acc = utils.accuracy3(pre_backdoor_label, backdoor_labels)

                backdoor_one_hot = label_to_onehot(backdoor_labels, num_classes=NUM_CLASSES)
                _, test_backdoor_one_hot = encoder(backdoor_one_hot)
                losses_backdoor = cross_entropy_for_one_hot(logits_backdoor, test_backdoor_one_hot)

                top1_backdoor = acc

            else:
                logits_backdoor = active_model(z_up, z_down)
                loss_backdoor = criterion(logits_backdoor, backdoor_labels)
                prec1 = utils.accuracy(logits_backdoor, backdoor_labels, topk=(1,))

                losses_backdoor = loss_backdoor.item()
                top1_backdoor = prec1[0]

            debug_log_list[2].append(z_up.cpu().numpy())
            debug_log_list[3].append(z_down.cpu().numpy())

        writer.add_scalar('val/loss', losses_valid.avg, cur_step)
        writer.add_scalar('val/top1_valid', top1_valid.avg, cur_step)
        writer.add_scalar('backdoor/loss', losses_backdoor, cur_step)
        writer.add_scalar('backdoor/top1_valid', top1_backdoor, cur_step)

        template = 'Epoch {}, Poisoned {}/{}, Loss: {:.4f}, Accuracy: {:.2f}, ' \
                   'Test Loss: {:.4f}, Test Accuracy: {:.2f}, ' \
                   'Backdoor Loss: {:.4f}, Backdoor Accuracy: {:.2f}\n'

        logging.info(template.format(epoch + 1,
                                     output_replace_count,
                                     gradient_replace_count,
                                     losses.avg,
                                     top1.avg,
                                     losses_valid.avg,
                                     top1_valid.avg,
                                     losses_backdoor,
                                     top1_backdoor.item(),
                                     ))

        if losses_valid.avg > 1e8 or np.isnan(losses_valid.avg):
            logging.info('********* INSTABLE TRAINING, BREAK **********')
            break

        valid_acc_top1 = top1_valid.avg
        # save
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
        logging.info('best_acc_top1 %f', best_acc_top1)

        # update scheduler
        for scheduler in scheduler_list:
            scheduler.step()

        with open('{}'.format(os.path.join(args.name, '_debug.pickle')), 'wb') as handle:
            pickle.dump(debug_log_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_poisoned_matrix(passive_matrix, need_poison, poison_grad, amplify_rate):
    poisoned_matrix = passive_matrix
    poisoned_matrix[need_poison] = poison_grad * amplify_rate
    return poisoned_matrix


def copy_grad(passive_matrix, need_copy):
    poison_grad = passive_matrix[need_copy]
    return poison_grad


if __name__ == '__main__':
    main()
