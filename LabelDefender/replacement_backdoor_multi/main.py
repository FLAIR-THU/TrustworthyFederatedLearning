import os
import sys
import numpy as np
import time

import random
import logging
import argparse
from numpy.core.numeric import require
import torch.nn as nn
from torch.types import Device
import torch.utils
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import utils
import copy

import torch

from dataset.mnist_dataset_vfl import MNISTDatasetVFL, need_poison_down_check_mnist_vfl
# from dataset.nuswide_dataset_vfl import NUSWIDEDatasetVFL, need_poison_down_check_nuswide_vfl

from dataset.cifar10_dataset_vfl import Cifar10DatasetVFL, need_poison_down_check_cifar10_vfl
from dataset.cifar100_dataset_vfl import Cifar100DatasetVFL, Cifar100DatasetVFL20Classes, \
    need_poison_down_check_cifar100_vfl

from models.model_templates import ClassificationModelGuest, ClassificationModelHostHead, \
    ClassificationModelHostTrainableHead, ClassificationModelHostHeadWithSoftmax,\
    SimpleCNN, LeNet5, MLP2
from models.resnet_torch import resnet18, resnet50


def main():
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--dataset', type=str, default='cifar20',
                        help='location of the data corpus')
    parser.add_argument('--name', type=str, default='test', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--workers', type=int, default=0, help='num of workers')
    parser.add_argument('--epochs', type=int, default=16, help='num of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--k', type=int, default=4, help='num of client')
    parser.add_argument('--model', default='mlp2', help='resnet')
    parser.add_argument('--input_size', type=int, default=28, help='resnet')
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--backdoor', type=int, default=0)
    parser.add_argument('--amplify_rate', type=float, default=10)
    parser.add_argument('--amplify_rate_output', type=float, default=1)
    parser.add_argument('--explicit_softmax', type=int, default=0)
    parser.add_argument('--random_output', type=int, default=0)
    parser.add_argument('--dp_type', type=str, default='none', help='[laplace, gaussian]')
    parser.add_argument('--dp_strength', type=float, default=0, help='[0.1, 0.075, 0.05, 0.025,...]')
    parser.add_argument('--gradient_sparsification', type=float, default=0)
    parser.add_argument('--writer', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--defense_up', type=int, default=1)
    
    parser.add_argument("--certify", type=int, default=0, help="CertifyFLBaseline")
    parser.add_argument("--M", type=int, default=1000, help="voting party count in CertifyFL")
    parser.add_argument("--sigma", type=float, default=0, help='sigma for certify')
    parser.add_argument("--adversarial_start_epoch", type=int, default=0, help="value of adversarial start epoch")
    parser.add_argument("--certify_start_epoch", type=int, default=1, help="number of epoch when the cerfity ClipAndPerturb start")
    
    parser.add_argument('--autoencoder', type=int, default=0)
    parser.add_argument('--lba', type=float, default=0)
    
    args = parser.parse_args()

    args.name = 'experiment_result_{}/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
        args.name, args.epochs, args.dataset, args.model, args.batch_size, args.name,args.backdoor, args.amplify_rate,
        args.amplify_rate_output, args.dp_type, args.dp_strength, args.gradient_sparsification, args.certify, args.sigma, args.autoencoder, args.lba, args.seed,
        args.use_project_head, args.random_output, args.learning_rate, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.name)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # tensorboard
    if args.writer == 1:
        writer = SummaryWriter(log_dir=os.path.join(args.name, 'tb'))
        writer.add_text('experiment', args.name, 0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = device

    logging.info('***** USED DEVICE: {}'.format(device))

    # set seed for target label and poisoned and target sample selection
    manual_seed = 42
    random.seed(manual_seed)

    # set seed for model initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print(args.gpu)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


    ##### set dataset
    input_dims = None

    if args.dataset == 'mnist':
        NUM_CLASSES = 10
        input_dims = [14 * 14, 14 * 14, 14 * 14, 14 * 14]
        args.input_size = 28 #TODO::??
        DATA_DIR = './dataset/MNIST'
        target_label = random.randint(0, NUM_CLASSES-1) # a class-id randomly created
        logging.info('target label: {}'.format(target_label))

        train_dataset = MNISTDatasetVFL(DATA_DIR, 'train', args.input_size, args.input_size, 600, 10, target_label)
        valid_dataset = MNISTDatasetVFL(DATA_DIR, 'test', args.input_size, args.input_size, 100, 10, target_label)

        # set poison_check function
        need_poison_down_check = need_poison_down_check_mnist_vfl

    # elif args.dataset == 'nuswide':
    #     NUM_CLASSES = 5
    #     input_dims = [634, 1000]
    #     DATA_DIR = './dataset/NUS_WIDE'
    #     target_label = 1 #random.sample([0,1,3,4], 1)[0]
    #     logging.info('target label: {}'.format(target_label))

    #     train_dataset = NUSWIDEDatasetVFL(DATA_DIR, 'train', 10, target_label)
    #     valid_dataset = NUSWIDEDatasetVFL(DATA_DIR, 'test', 10, target_label)

    #     # set poison_check function
    #     need_poison_down_check = need_poison_down_check_nuswide_vfl

    elif args.dataset == 'cifar10':
        NUM_CLASSES = 10
        input_dims = [16 * 16, 16 * 16, 16 * 16, 16 * 16]
        args.input_size = 32

        DATA_DIR = './dataset/cifar-10-batches-py'

        target_label = random.randint(0, NUM_CLASSES-1)
        logging.info('target label: {}'.format(target_label))

        train_dataset = Cifar10DatasetVFL(DATA_DIR, 'train', args.input_size, args.input_size, 500, 10, target_label)
        valid_dataset = Cifar10DatasetVFL(DATA_DIR, 'test', args.input_size, args.input_size, 100, 10, target_label)

        # set poison_check function
        need_poison_down_check = need_poison_down_check_cifar10_vfl

    elif args.dataset == 'cifar20':
        NUM_CLASSES = 20
        input_dims = [16 * 16, 16 * 16, 16 * 16, 16 * 16]
        args.input_size = 32
        DATA_DIR = './dataset/cifar-100-python'
        target_label = random.randint(0, NUM_CLASSES-1)
        logging.info('target label: {}'.format(target_label))

        train_dataset = Cifar100DatasetVFL20Classes(DATA_DIR, 'train', args.input_size, args.input_size, 200, 10, target_label)
        valid_dataset = Cifar100DatasetVFL20Classes(DATA_DIR, 'test', args.input_size, args.input_size, 20, 10, target_label)

        # set poison_check function
        need_poison_down_check = need_poison_down_check_cifar100_vfl

    n_train = len(train_dataset)
    n_valid = len(valid_dataset)

    # print(len(train_dataset.x),train_dataset.x[3].shape,n_train,n_valid)

    train_indices = list(range(n_train))
    valid_indices = list(range(n_valid))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.workers, # number of child process, multi-process
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    # check poisoned samples
    print('train poison samples:', sum(need_poison_down_check(train_dataset.x))) # should be zero, but is that right in the belowing?
    print('test poison samples:', sum(need_poison_down_check(valid_dataset.x)))
    print(train_dataset.poison_list[:10])
    poison_list = train_dataset.poison_list

    ##### set model
    local_models = []
    if args.model == 'mlp2':
        for i in range(args.k):
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
        for i in range(args.k):
            backbone = LeNet5(NUM_CLASSES)
            local_models.append(backbone)

    criterion = nn.CrossEntropyLoss()

    model_list = []
    for i in range(args.k+1):
        if i == 0:
            if args.use_project_head == 1:
                active_model = ClassificationModelHostTrainableHead(NUM_CLASSES*args.k, NUM_CLASSES).to(device)
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
            model_list.append(ClassificationModelGuest(local_models[i-1]))

    local_models = None
    model_list = [model.to(device) for model in model_list]

    criterion = criterion.to(device)

    # weights optimizer
    optimizer_active_model = None
    optimizer_list = []
    if args.use_project_head == 1:
        optimizer_active_model = torch.optim.SGD(active_model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            for model in model_list]
    else:
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)
            for model in model_list]

    scheduler_list = []
    if args.learning_rate == 0.025:
        if optimizer_active_model is not None:
            scheduler_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_active_model, float(args.epochs)))
        scheduler_list = scheduler_list + [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
            for optimizer in optimizer_list]
    else:
        if optimizer_active_model is not None:
            scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_active_model, args.decay_period, gamma=args.gamma))
        scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                          in optimizer_list]

    assert(len(model_list)==args.k)
    assert(len(optimizer_list)==args.k)
    assert(len(scheduler_list)==args.k)

    best_acc_top1 = 0.

    # get train backdoor data
    train_backdoor_images, train_backdoor_true_labels = train_dataset.get_poison_data()

    # get train target data
    train_target_images, train_target_labels = train_dataset.get_target_data()
    print('train target data', train_target_images[0].shape, train_target_labels)
    print('train poison samples:', sum(need_poison_down_check(train_backdoor_images))) # should not be zero when using backdoor

    # get test backdoor data
    test_backdoor_images, test_backdoor_true_labels = valid_dataset.get_poison_data()

    # set test backdoor label
    test_backdoor_labels = copy.deepcopy(test_backdoor_true_labels)
    test_backdoor_labels[:] = valid_dataset.target_label

    target_label = train_dataset.target_label
    print('the label of the sample need copy = ', train_dataset.target_label, valid_dataset.target_label)


    amplify_rate = torch.tensor(args.amplify_rate).float().to(device)

    # active_up_gradients_res = None
    # active_down_gradients_res = None
    active_gradients_res_list = [None for _ in range(args.k)]
    
    # loop
    for epoch in range(args.epochs):

        output_replace_count = 0
        gradient_replace_count = 0
        ########### TRAIN ###########
        top1 = utils.AverageMeter()
        losses = utils.AverageMeter()

        cur_step = epoch * len(train_loader)
        cur_lr = optimizer_list[0].param_groups[0]['lr']
        if args.writer == 1:
            writer.add_scalar('train/lr', cur_lr, cur_step)

        # if args.certify != 0 and epoch >= args.certify_start_epoch:
        #     print("ClipAndPerturb the model in the training process")
        #     for ik in range(args.k):
        #         model_list[ik] = utils.ClipAndPerturb(model_list[ik],device,epoch*0.1+2,args.sigma)
        
        for model in model_list:
            active_model.train()
            model.train()

        for step, (trn_X, trn_y) in enumerate(train_loader):

            # select one backdoor data
            id = random.randint(0, train_backdoor_images[0].shape[0]-1)
            backdoor_image_list = [train_backdoor_images[il][id] for il in range(args.k)]
            # backdoor_image_up = train_backdoor_images[0][id]
            # backdoor_image_down = train_backdoor_images[1][id]
            backdoor_label = train_backdoor_true_labels[id]
            # select one target data
            id = random.randint(0, train_target_images[0].shape[0]-1)
            target_image_list = [train_target_images[il][id] for il in range(args.k)]
            # target_image_up = train_target_images[0][id]
            # target_image_down = train_target_images[1][id]

            # merge normal train data with selected backdoor and target data
            trn_X_list = []
            for i in range(args.k):
                trn_X_list.append(np.concatenate([trn_X[i].numpy(), np.expand_dims(backdoor_image_list[i], 0), np.expand_dims(target_image_list[i],0)]))
            # trn_X_up = np.concatenate([trn_X[0].numpy(), np.expand_dims(backdoor_image_up, 0), np.expand_dims(target_image_up,0)])
            # trn_X_down = np.concatenate([trn_X[1].numpy(), np.expand_dims(backdoor_image_down, 0), np.expand_dims(target_image_down,0)])
            trn_y = np.concatenate([trn_y.numpy(), np.array([[backdoor_label]]), np.array([[target_label]])])

            for i in range(args.k):
                trn_X_list[i] = torch.from_numpy(trn_X_list[i]).float().to(device)
            # trn_X_up = torch.from_numpy(trn_X_up).float().to(device)
            # trn_X_down = torch.from_numpy(trn_X_down).float().to(device)
            target = torch.from_numpy(trn_y).view(-1).long().to(device)

            N = target.size(0)

            # passive party 0~3 generate output
            z_list = []
            z_list_clone = []
            for i in range(args.k):
                z_list.append(model_list[i](trn_X_list[i]))
                if args.certify != 0 and epoch >= args.certify_start_epoch and i != 0:
                    z_list[i] = utils.ClipAndPerturb(z_list[i],device,epoch*0.1+2,args.sigma)
                z_list_clone.append(z_list[i].detach().clone())

            # # passive party 0 generate output
            # z_up = model_list[0](trn_X_up)
            # z_up_clone = z_up.detach().clone()
            # z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(args.device)

            # # passive party 1 generate output
            # z_down = model_list[1](trn_X_down)
            # z_down_clone = z_down.detach().clone()

            ########### backdoor: replace output of target data ##########
            if args.backdoor == 1 and epoch >= args.adversarial_start_epoch:
                with torch.no_grad():
                    for i in range(args.k-1):
                        # attack are 1,2,...,k-1; active party is 0
                        z_list_clone[i+1][-1] = z_list_clone[i+1][-2] # replace target data output using poisoned data output
                    output_replace_count = output_replace_count + 1
             ########### backdoor end here ##########

            for i in range(args.k):
                z_list_clone[i] = torch.autograd.Variable(z_list_clone[i], requires_grad=True).to(args.device)
            # z_down_clone = torch.autograd.Variable(z_down_clone, requires_grad=True).to(args.device)

            # active party backward
            logits = active_model(z_list_clone)
            # logits = active_model(z_up_clone, z_down_clone)
            loss = criterion(logits, target)

            # z_gradients_up = torch.autograd.grad(loss, z_up_clone, retain_graph=True)
            # z_gradients_down = torch.autograd.grad(loss, z_down_clone, retain_graph=True)
            z_gradients_list = [torch.autograd.grad(loss, z_list_clone[i], retain_graph=True) for i in range(args.k)]

            # z_gradients_up_clone = z_gradients_up[0].detach().clone()
            # z_gradients_down_clone = z_gradients_down[0].detach().clone()
            z_gradients_list_clone = [(z_gradients_list[i][0].detach().clone()) for i in range(args.k)]

            # update active model
            if optimizer_active_model is not None:
                optimizer_active_model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_active_model.step()

            ########### defense start here ##########
            location = 0.0
            threshold = 0.2
            if args.dp_type == 'laplace':
                with torch.no_grad():
                    scale = args.dp_strength
                    # clip 2-norm per sample
                    # norm_factor_up = torch.div(torch.max(torch.norm(z_gradients_up_clone, dim=1)),
                    #                            threshold + 1e-6).clamp(min=1.0)
                    # norm_factor_down = torch.div(torch.max(torch.norm(z_gradients_down_clone, dim=1)),
                    #                            threshold + 1e-6).clamp(min=1.0)
                    norm_factor_list = [(torch.div(torch.max(torch.norm(z_gradients_list_clone[i], dim=1)),
                                               threshold + 1e-6).clamp(min=1.0)) for i in range(args.k)]

                    # add laplace noise
                    # dist_up = torch.distributions.laplace.Laplace(location, scale)
                    # dist_down = torch.distributions.laplace.Laplace(location, scale)
                    dist_list = [torch.distributions.laplace.Laplace(location, scale) for _ in range(args.k)]

                    # if args.defense_up == 1:
                    #     z_gradients_up_clone = torch.div(z_gradients_up_clone, norm_factor_up) + \
                    #                            dist_up.sample(z_gradients_up_clone.shape).to(device)
                    # z_gradients_down_clone = torch.div(z_gradients_down_clone, norm_factor_down) + \
                    #                          dist_down.sample(z_gradients_down_clone.shape).to(device)
                    # since we alwasys have args.defense_up==1
                    z_gradients_list_clone = [torch.div(z_gradients_list_clone[i], norm_factor_list[i]) + \
                                             dist_list[i].sample(z_gradients_list_clone[i].shape).to(device) for i in range(args.k)]
                    # z_gradients_list_clone.to(device)

            if args.dp_type == 'gaussian':
                with torch.no_grad():
                    scale = args.dp_strength
                    # norm_factor_up = torch.div(torch.max(torch.norm(z_gradients_up_clone, dim=1)),
                    #                            threshold + 1e-6).clamp(min=1.0)
                    # norm_factor_down = torch.div(torch.max(torch.norm(z_gradients_down_clone, dim=1)),
                    #                            threshold + 1e-6).clamp(min=1.0)
                    norm_factor_list = [(torch.div(torch.max(torch.norm(z_gradients_list_clone[i], dim=1)),
                                               threshold + 1e-6).clamp(min=1.0)) for i in range(args.k)]
                    # if args.defense_up == 1:
                    #     z_gradients_up_clone = torch.div(z_gradients_up_clone, norm_factor_up) + \
                    #                            torch.normal(location, scale, z_gradients_up_clone.shape).to(device)
                    # z_gradients_down_clone = torch.div(z_gradients_down_clone, norm_factor_down) + \
                    #                          torch.normal(location, scale, z_gradients_down_clone.shape).to(device)
                    # since we alwasys have args.defense_up==1
                    z_gradients_list_clone = [torch.div(z_gradients_list_clone[i], norm_factor_list[i]) + \
                                             torch.normal(location, scale, z_gradients_list_clone[i].shape).to(device) for i in range(args.k)]
                    # z_gradients_list_clone.to(device)
            if args.gradient_sparsification != 0:
                with torch.no_grad():
                    percent = args.gradient_sparsification / 100.0
                    # if active_up_gradients_res is not None and \
                    #         z_gradients_up_clone.shape[0] == active_up_gradients_res.shape[0] and args.defense_up == 1:
                    #     z_gradients_up_clone = z_gradients_up_clone + active_up_gradients_res
                    # if active_down_gradients_res is not None and \
                    #         z_gradients_down_clone.shape[0] == active_down_gradients_res.shape[0]:
                    #     z_gradients_down_clone = z_gradients_down_clone + active_down_gradients_res
                    for i in range(args.k):
                        if active_gradients_res_list[i] is not None and \
                                z_gradients_list_clone[i].shape[0] == active_gradients_res_list[i].shape[0] and args.defense_up == 1:
                            z_gradients_list_clone[i] = z_gradients_list_clone[i] + active_gradients_res_list[i]
                    
                    # up_thr = torch.quantile(torch.abs(z_gradients_up_clone), percent)
                    # down_thr = torch.quantile(torch.abs(z_gradients_down_clone), percent)
                    thr_list = []
                    for i in range(args.k):
                        thr_list.append(torch.quantile(torch.abs(z_gradients_list_clone[i]), percent))
                    
                    # active_up_gradients_res = torch.where(torch.abs(z_gradients_up_clone).double() < up_thr.item(),
                    #                                       z_gradients_up_clone.double(), float(0.)).to(device)
                    # active_down_gradients_res = torch.where(torch.abs(z_gradients_down_clone).double() < down_thr.item(),
                    #                                       z_gradients_down_clone.double(), float(0.)).to(device)
                    for i in range(args.k):
                        active_gradients_res_list[i] = torch.where(torch.abs(z_gradients_list_clone[i]).double() < thr_list[i].item(),
                                                          z_gradients_list_clone[i].double(), float(0.)).to(device)

                    # if args.defense_up == 1:
                    #     z_gradients_up_clone = z_gradients_up_clone - active_up_gradients_res
                    # z_gradients_down_clone = z_gradients_down_clone - active_down_gradients_res
                    # since we alwasys have args.defense_up==1
                    z_gradients_list_clone = [(z_gradients_list_clone[i] - active_gradients_res_list[i]) for i in range(args.k)]
                    # z_gradients_list_clone.to(device)
            ########### defense end here ##########


            ########### backdoor: replace gradient for poisoned data ##########
            if args.backdoor == 1:
                with torch.no_grad():
                    for i in range(args.k-1):
                        z_gradients_list_clone[i+1][-2] = z_gradients_list_clone[i+1][-1]*amplify_rate # replace the received poisoned gradient using target gradient,contradict with the paper??? 
                    gradient_replace_count = gradient_replace_count + 1
            ########### backdoor end here ##########

            # update passive model 0
            optimizer_list[0].zero_grad()
            weights_gradients_list = []
            weights_gradients_list.append(torch.autograd.grad(z_list[0], model_list[0].parameters(),
                                                    grad_outputs=z_gradients_list_clone[0]))

            for w, g in zip(model_list[0].parameters(), weights_gradients_list[0]):
                if w.requires_grad:
                    w.grad = g.detach()
            optimizer_list[0].step()

            # update passive model 1~3
            # optimizer_list[1].zero_grad()
            # if args.backdoor == 1:
            #     weights_gradients_down = torch.autograd.grad(z_down[:-1], model_list[1].parameters(),
            #                                                  grad_outputs=z_gradients_down_clone[:-1])
            # else:
            #     weights_gradients_down = torch.autograd.grad(z_down, model_list[1].parameters(),
            #                                                  grad_outputs=z_gradients_down_clone)
            # for w, g in zip(model_list[1].parameters(), weights_gradients_down):
            #     if w.requires_grad:
            #         w.grad = g.detach()
            # optimizer_list[1].step()
            for i in range(args.k-1):
                optimizer_list[i+1].zero_grad()
                if args.backdoor == 1:
                    weights_gradients_list.append(torch.autograd.grad(z_list[i+1][:-1], model_list[i+1].parameters(),
                                                                        grad_outputs=z_gradients_list_clone[i+1][:-1]))
                else:
                    weights_gradients_list.append(torch.autograd.grad(z_list[i+1], model_list[i+1].parameters(),
                                                                        grad_outputs=z_gradients_list_clone[i+1]))
                for w,g in zip(model_list[i+1].parameters(), weights_gradients_list[i+1]):
                    if w.requires_grad:
                        w.grad = g.detach()
                optimizer_list[i+1].step()


            # train metrics
            prec1 = utils.accuracy(logits, target, topk=(1,))
            losses.update(loss.item(), N)
            top1.update(prec1[0].item(), N)
            if args.writer == 1:
                writer.add_scalar('train/loss', losses.avg, cur_step)
                writer.add_scalar('train/top1', top1.avg, cur_step)
            cur_step += 1

        cur_step = (epoch + 1) * len(train_loader)

        ########### VALIDATION ###########
        top1_valid = utils.AverageMeter()
        losses_valid = utils.AverageMeter()

        # validate_model_list = []
        for model in model_list:
            active_model.eval()
            model.eval()
        
        if args.certify != 0 and epoch >= args.certify_start_epoch:
            # for m in range(args.M):
            #     validate_model_list.append([utils.ClipAndPerturb(model_list[i],device,args.epochs*0.1+2,args.sigma) for i in range(args.k)])
            # for m in range(args.M):
            #     active_model.eval()
            #     for i in range(args.k):
            #         validate_model_list[m][i].eval()


            with torch.no_grad():
                # test accuracy
                for step, (val_X, val_y) in enumerate(valid_loader):
                    val_X = [x.float().to(args.device) for x in val_X]
                    target = val_y.view(-1).long().to(args.device)
                    N = target.size(0)
                
                    loss_list = []
                    logits_list = []
                    vote_list = []
                    z_list = [model_list[0](val_X[0])]
                    for m in range(args.M):
                        # z_up = validate_model_list[m][0](val_X[0])
                        # z_down = validate_model_list[m][1](val_X[1])
                        z_list = z_list[:1]
                        for i in range(args.k-1):
                            z_list.append(model_list[i+1](val_X[i+1]))
                            z_list[i+1] = utils.ClipAndPerturb(z_list[i+1],device,args.epochs*0.1+2,args.sigma)

                        # logits = active_model(z_up, z_down)
                        logits = active_model(z_list)
                        loss = criterion(logits, target)
                        logits_list.append(logits) #logits.shape=[64,10]
                        loss_list.append(loss)

                        vote = utils.vote(logits,topk=(1,))
                        vote_list.append(vote.cpu().numpy())
                    #vote_list.shape=(m,N) in cpu as numpy.array
                    # print("N=",N,"vote_list.shape=",(np.asarray(vote_list)).shape) #N=64, vote_list.shape=(m,N)
                    vote_list = np.transpose(np.asarray(vote_list))
                    vote_result = np.apply_along_axis(lambda x: np.bincount(x, minlength=logits.shape[1]), axis=1, arr=vote_list)
                    # print(vote_result.shape) #(N,logits.shape[1]=#class)
                    vote_result = torch.from_numpy(vote_result)
                    vote_result.to(device)
                    
                    prec1 = utils.accuracy2(vote_result, target, args.M, device, topk=(1,2,))
                    
                    losses_valid.update(loss.item(), N)
                    top1_valid.update(prec1[0].item(), N)

                # backdoor related metrics
                # backdoor_X_up = torch.from_numpy(test_backdoor_images[0]).float().to(args.device)
                # backdoor_X_down = torch.from_numpy(test_backdoor_images[1]).float().to(args.device)
                backdoor_X_list = [torch.from_numpy(test_backdoor_images[i]).float().to(args.device) for i in range(args.k)]
                backdoor_labels = torch.from_numpy(test_backdoor_labels).long().to(args.device)

                N = backdoor_labels.shape[0]
                
                loss_list = []
                logits_list = []
                vote_list = []
                z_list = [model_list[0](backdoor_X_list[0])]
                for m in range(args.M):
                    # z_up = validate_model_list[m][0](backdoor_X_up)
                    # z_down = validate_model_list[m][1](backdoor_X_down)
                    z_list = z_list[:1]
                    for i in range(args.k-1):
                        z_list.append(model_list[i+1](backdoor_X_list[i+1]))
                        z_list[i+1] = utils.ClipAndPerturb(z_list[i+1],device,args.epochs*0.1+2,args.sigma)

                    ########## backdoor metric
                    # logits_backdoor = active_model(z_up, z_down)
                    logits_backdoor = active_model(z_list)
                    loss_backdoor = criterion(logits_backdoor, backdoor_labels)
                    logits_list.append(logits_backdoor)
                    loss_list.append(loss_backdoor)

                    vote = utils.vote(logits_backdoor,topk=(1,))
                    vote_list.append(vote.cpu().numpy())
                #vote_list.shape=(m,N) in cpu as numpy.array
                # print("N=",N,"vote_list.shape=",(np.asarray(vote_list)).shape) #N=64, vote_list.shape=(m,N)
                vote_list = np.transpose(np.asarray(vote_list))
                # print("vote_list.shape =",vote_list.shape, " logits.shape =", logits_backdoor.shape, "N =",N)
                vote_result = np.apply_along_axis(lambda x: np.bincount(x, minlength=logits.shape[1]), axis=1, arr=vote_list)
                # print(vote_result.shape) #(N,logits.shape[1]=#class)
                vote_result = torch.from_numpy(vote_result)
                vote_result.to(device)
                
                prec1 = utils.accuracy2(vote_result, backdoor_labels[0:vote_list.shape[0]], args.M, device, topk=(1,2,))

                losses_backdoor = loss_backdoor.item()
                top1_backdoor = prec1[0]
        else:


            with torch.no_grad():
                # test accuracy
                for step, (val_X, val_y) in enumerate(valid_loader):
                    val_X = [x.float().to(args.device) for x in val_X]
                    target = val_y.view(-1).long().to(args.device)
                    N = target.size(0)

                    # z_up = model_list[0](val_X[0])
                    # z_down = model_list[1](val_X[1])
                    z_list = [model_list[i](val_X[i]) for i in range(args.k)]

                    # logits = active_model(z_up, z_down)
                    logits = active_model(z_list)
                    loss = criterion(logits, target)

                    prec1 = utils.accuracy(logits, target, topk=(1,))

                    losses_valid.update(loss.item(), N)
                    top1_valid.update(prec1[0].item(), N)

                # backdoor related metrics
                # backdoor_X_up = torch.from_numpy(test_backdoor_images[0]).float().to(args.device)
                # backdoor_X_down = torch.from_numpy(test_backdoor_images[1]).float().to(args.device)
                backdoor_X_list = [torch.from_numpy(test_backdoor_images[i]).float().to(args.device) for i in range(args.k)]
                backdoor_labels = torch.from_numpy(test_backdoor_labels).long().to(args.device)

                N = backdoor_labels.shape[0]

                # z_up = model_list[0](backdoor_X_up)
                # z_down = model_list[1](backdoor_X_down)
                z_list = [model_list[i](backdoor_X_list[i]) for i in range(args.k)]

                ########## backdoor metric
                logits_backdoor = active_model(z_list)
                loss_backdoor = criterion(logits_backdoor, backdoor_labels)

                prec1 = utils.accuracy(logits_backdoor, backdoor_labels, topk=(1,))

                losses_backdoor = loss_backdoor.item()
                top1_backdoor = prec1[0]

        if args.writer == 1:
            writer.add_scalar('val/loss', losses_valid.avg, cur_step)
            writer.add_scalar('val/top1_valid', top1_valid.avg, cur_step)
            writer.add_scalar('backdoor/loss', losses_backdoor, cur_step)
            writer.add_scalar('backdoor/top1_valid', top1_backdoor, cur_step)


        template = 'Epoch {}, Poisoned {}/{}, Loss: {:.4f}, Accuracy: {:.2f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}, ' \
                   'Backdoor Loss: {:.4f}, Backdoor Accuracy: {:.2f}\n ' \

        logging.info(template.format(epoch + 1,
                      output_replace_count,
                      gradient_replace_count,
                      losses.avg,
                      top1.avg,
                      losses_valid.avg,
                      top1_valid.avg,
                      losses_backdoor,
                      top1_backdoor.item()))

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


def get_poisoned_matrix(passive_matrix, need_poison, poison_grad, amplify_rate):
    poisoned_matrix = passive_matrix
    poisoned_matrix[need_poison] = poison_grad * amplify_rate
    return poisoned_matrix


def copy_grad(passive_matrix, need_copy):
    poison_grad = passive_matrix[need_copy]
    return poison_grad


if __name__ == '__main__':
    main()
