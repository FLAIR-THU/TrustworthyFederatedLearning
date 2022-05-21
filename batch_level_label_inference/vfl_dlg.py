import logging
import pprint
import time

import tensorflow as tf
import marvell_shared_variable as shared_var
from marvell_solver import solve_isotropic_covariance, symKL_objective

import torch
import torch.nn.functional as F
import numpy as np

from models.vision import *
from utils import *
from utils_mc_defense import *


tf.compat.v1.enable_eager_execution() 


# function for marvell noise strength calculation
def KL_gradient_perturb(g, classes, sumKL_threshold, dynamic=False, init_scale=1.0, uv_choice='uv', p_frac='pos_frac'):
    assert len(classes)==2

    # g is a torch.tensor
    # Torch => numpy
    numpy_g = g.cpu().numpy()
    # numpy => Tensorflow
    g = tf.convert_to_tensor(numpy_g)

    g_original_shape = g.shape
    g = tf.reshape(g, shape=(g_original_shape[0], -1))
    # print(g)

    _y = shared_var.batch_y
    # y = y.cpu().numpy()
    y = _y
    # y = tf.as_dtype(_y)
    pos_g = g[y==1]
    pos_g_mean = tf.math.reduce_mean(pos_g, axis=0, keepdims=True) # shape [1, d]
    pos_coordinate_var = tf.reduce_mean(tf.math.square(pos_g - pos_g_mean), axis=0) # use broadcast
    neg_g = g[y==0]
    neg_g_mean = tf.math.reduce_mean(neg_g, axis=0, keepdims=True) # shape [1, d]
    neg_coordinate_var = tf.reduce_mean(tf.math.square(neg_g - neg_g_mean), axis=0)

    avg_pos_coordinate_var = tf.reduce_mean(pos_coordinate_var)
    avg_neg_coordinate_var = tf.reduce_mean(neg_coordinate_var)

    g_diff = pos_g_mean - neg_g_mean
    g_diff_norm = float(tf.norm(tensor=g_diff).numpy())
    # if g_diff_norm ** 2 > 1:
    #     print('pos_g_mean', pos_g_mean.shape)
    #     print('neg_g_mean', neg_g_mean.shape)
    #     assert g_diff_norm

    if uv_choice == 'uv':
        u = float(avg_neg_coordinate_var)
        v = float(avg_pos_coordinate_var)
        if u == 0.0:
            print('neg_g')
            print(neg_g)
        if v == 0.0:
            print('pos_g')
            print(pos_g)
    elif uv_choice == 'same':
        u = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
        v = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
    elif uv_choice == 'zero':
        u, v = 0.0, 0.0

    d = float(g.shape[1])

    if p_frac == 'pos_frac':
        p = float(tf.reduce_sum(y) / len(y)) # p is set as the fraction of positive in the batch
    else:
        p = float(p_frac)

    scale = init_scale
    P = scale * g_diff_norm**2
    # print('u={0},v={1},d={2},g={3},p={4},P={5}'.format(u,v,d,g_diff_norm**2,p,P))


    # print('compute problem instance', time.time() - start)
    # start = time.time()

    lam10, lam20, lam11, lam21 = None, None, None, None
    while True:
        P = scale * g_diff_norm**2
        # print('g_diff_norm ** 2', g_diff_norm ** 2)
        # print('P', P)
        # print('u, v, d, p', u, v, d, p)
        lam10, lam20, lam11, lam21, sumKL = \
            solve_isotropic_covariance(
                u=u,
                v=v,
                d=d,
                g=g_diff_norm ** 2,
                p=p,
                P=P,
                lam10_init=lam10,
                lam20_init=lam20,
                lam11_init=lam11,
                lam21_init=lam21)
        # print('sumKL', sumKL)
        # print()

        # print(scale)
        if not dynamic or sumKL <= sumKL_threshold:
            break

        scale *= 1.5 # loosen the power constraint
    
    # print('solving time', time.time() - start)
    # start = time.time()

    with shared_var.writer.as_default():
        tf.summary.scalar(name='solver/u',
                        data=u,
                        step=shared_var.counter)
        tf.summary.scalar(name='solver/v',
                        data=v,
                        step=shared_var.counter)
        tf.summary.scalar(name='solver/g',
                        data=g_diff_norm ** 2,
                        step=shared_var.counter)
        tf.summary.scalar(name='solver/p',
                        data=p,
                        step=shared_var.counter)
        tf.summary.scalar(name='solver/scale',
                            data=scale,
                            step=shared_var.counter)
        tf.summary.scalar(name='solver/P',
                        data=P,
                        step=shared_var.counter)
        tf.summary.scalar(name='solver/lam10',
                            data=lam10,
                            step=shared_var.counter)
        tf.summary.scalar(name='solver/lam20',
                            data=lam20,
                            step=shared_var.counter)
        tf.summary.scalar(name='solver/lam11',
                            data=lam11,
                            step=shared_var.counter)
        tf.summary.scalar(name='solver/lam21',
                            data=lam21,
                            step=shared_var.counter)
        # tf.summary.scalar(name='sumKL_before',
        #                 data=symKL_objective(lam10=0.0,lam20=0.0,lam11=0.0,lam21=0.0,
        #                                     u=u, v=v, d=d, g=g_diff_norm**2),
        #                 step=shared_var.counter)
        # even if we didn't use avg_neg_coordinate_var for u and avg_pos_coordinate_var for v, we use it to evaluate the sumKL_before
        tf.summary.scalar(name='sumKL_before',
                        data=symKL_objective(lam10=0.0,lam20=0.0,lam11=0.0,lam21=0.0,
                                            u=float(avg_neg_coordinate_var),
                                            v=float(avg_pos_coordinate_var),
                                            d=d, g=g_diff_norm**2),
                        step=shared_var.counter)
        tf.summary.scalar(name='sumKL_after',
                        data=sumKL,
                        step=shared_var.counter)
        tf.summary.scalar(name='error prob lower bound',
                        data=0.5 - math.sqrt(sumKL) / 4,
                        step=shared_var.counter)
    

    # print('tb logging', time.time() - start)
    # start = time.time()

    perturbed_g = g
    y_float = tf.cast(y, dtype=tf.float32)

    # positive examples add noise in g1 - g0
    perturbed_g += tf.reshape(tf.multiply(x=tf.random.normal(shape=y.shape),
                            y=y_float), shape=(-1, 1)) * g_diff * (math.sqrt(lam11-lam21)/g_diff_norm)

    # add spherical noise to positive examples
    if lam21 > 0.0:
        perturbed_g += tf.random.normal(shape=g.shape) * tf.reshape(y_float, shape=(-1, 1)) * math.sqrt(lam21)

    # negative examples add noise in g1 - g0
    perturbed_g += tf.reshape(tf.multiply(x=tf.random.normal(shape=y.shape),
                            y=1-y_float), shape=(-1, 1)) * g_diff * (math.sqrt(lam10-lam20)/g_diff_norm)

    # add spherical noise to negative examples
    if lam20 > 0.0:
        perturbed_g += tf.random.normal(shape=g.shape) * tf.reshape(1-y_float, shape=(-1, 1)) * math.sqrt(lam20)

    tf_tensor_result = tf.reshape(perturbed_g, shape=g_original_shape)
    # Tensorflow => Numpy
    numpy_result = tf_tensor_result.numpy()
    # Numpy => Torch
    torch_result = torch.from_numpy(numpy_result)
    return torch_result


class LabelLeakage(object):
    def __init__(self, args):
        '''
        :param args:  contains all the necessary parameters
        '''
        self.dataset = args.dataset
        self.model = args.model
        self.num_exp = args.num_exp
        self.epochs = args.epochs
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.early_stop_param = args.early_stop_param
        self.device = args.device
        self.batch_size_list = args.batch_size_list
        self.num_class_list = args.num_class_list
        self.dst = args.dst
        self.exp_res_dir = args.exp_res_dir
        self.exp_res_path = args.exp_res_path
        self.apply_trainable_layer = args.apply_trainable_layer
        self.apply_laplace = args.apply_laplace
        self.apply_gaussian = args.apply_gaussian
        self.dp_strength = args.dp_strength
        self.apply_grad_spar = args.apply_grad_spar
        self.grad_spars = args.grad_spars
        self.apply_encoder = args.apply_encoder
        self.apply_random_encoder = args.apply_random_encoder
        self.apply_adversarial_encoder = args.apply_adversarial_encoder
        self.ae_lambda = args.ae_lambda
        self.encoder = args.encoder
        self.apply_marvell = args.apply_marvell
        self.marvell_s = args.marvell_s
        self.apply_ppdl = args.apply_ppdl
        self.ppdl_theta_u = args.ppdl_theta_u
        self.apply_gc = args.apply_gc
        self.gc_preserved_percent = args.gc_preserved_percent
        self.apply_lap_noise = args.apply_lap_noise
        self.noise_scale = args.noise_scale
        self.apply_discrete_gradients = args.apply_discrete_gradients
        self.discrete_gradients_bins = args.discrete_gradients_bins
        self.discrete_gradients_bound = args.discrete_gradients_bound

        self.show_param()

    def show_param(self):
        print(f'********** config dict **********')
        pprint.pprint(self.__dict__)

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def get_random_softmax_onehot_label(self, gt_onehot_label):
        _random = torch.randn(gt_onehot_label.size()).to(self.device)
        for i in range(len(gt_onehot_label)):
            max_index, = torch.where(_random[i] == _random[i].max())
            max_label, = torch.where(gt_onehot_label[i] == gt_onehot_label[i].max())
            while len(max_index) > 1:
                temp = torch.randn(gt_onehot_label[i].size()).to(self.device)
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
        return self.encoder(_random)

    def train(self):
        '''
        execute the label inference algorithm
        :return: recovery rate
        '''

        print(f"Running on %s{torch.cuda.current_device()}" % self.device)
        if self.dataset == 'nuswide':
            all_nuswide_labels = []
            for line in os.listdir('./data/NUS_WIDE/Groundtruth/AllLabels'):
                all_nuswide_labels.append(line.split('_')[1][:-4])
        for batch_size in self.batch_size_list:
            for num_classes in self.num_class_list:
                classes = [None] * num_classes
                if self.dataset == 'cifar100':
                    classes = random.sample(list(range(100)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)
                elif self.dataset == 'mnist':
                    classes = random.sample(list(range(10)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)
                elif self.dataset == 'nuswide':
                    classes = random.sample(all_nuswide_labels, num_classes)
                    x_image, x_text, Y = get_labeled_data('./data/NUS_WIDE', classes, None, 'Train')
                elif self.dataset == 'cifar10':
                    classes = random.sample(list(range(10)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)

                recovery_rate_history = []
                for i_run in range(1, self.num_exp + 1):
                    start_time = time.time()
                    # randomly sample
                    if self.dataset == 'mnist' or self.dataset == 'cifar100' or self.dataset == 'cifar10':
                        gt_data = []
                        gt_label = []
                        for i in range(0, batch_size):
                            sample_idx = torch.randint(len(all_data), size=(1,)).item()
                            gt_data.append(all_data[sample_idx])
                            gt_label.append(all_label[sample_idx])
                        gt_data = torch.stack(gt_data).to(self.device)
                        half_size = list(gt_data.size())[-1] // 2
                        gt_data_a = gt_data[:, :, :half_size, :]
                        gt_data_b = gt_data[:, :, half_size:, :]
                        gt_label = torch.stack(gt_label).to(self.device)
                        gt_onehot_label = gt_label  # label_to_onehot(gt_label)
                    elif self.dataset == 'nuswide':
                        gt_data_a, gt_data_b, gt_label = [], [], []
                        for i in range(0, batch_size):
                            sample_idx = torch.randint(len(x_image), size=(1,)).item()
                            gt_data_a.append(torch.tensor(x_text[sample_idx], dtype=torch.float32))
                            gt_data_b.append(torch.tensor(x_image[sample_idx], dtype=torch.float32))
                            gt_label.append(torch.tensor(Y[sample_idx], dtype=torch.float32))
                        gt_data_a = torch.stack(gt_data_a).to(self.device)
                        gt_data_b = torch.stack(gt_data_b).to(self.device)
                        gt_label = torch.stack(gt_label).to(self.device)
                        gt_onehot_label = gt_label  # label_to_onehot(gt_label)
                    if self.apply_encoder:
                        if not self.apply_random_encoder:
                            _, gt_onehot_label = self.encoder(gt_onehot_label) # get the result given by AutoEncoder.forward
                        else:
                            _, gt_onehot_label = self.get_random_softmax_onehot_label(gt_onehot_label)
                    if self.apply_adversarial_encoder:
                        _, gt_onehot_label = self.encoder(gt_data_a)
                    # set model
                    if self.model == 'MLP2':
                        net_a = MLP2(np.prod(list(gt_data_a.size())[1:]), num_classes).to(self.device)
                        net_b = MLP2(np.prod(list(gt_data_b.size())[1:]), num_classes).to(self.device)
                    elif self.model == 'resnet18':
                        net_a = resnet18(num_classes).to(self.device)
                        net_b = resnet18(num_classes).to(self.device)
                    
                    # ......if args.apply_certify != 0 and epoch >= args.certify_start_epoch:
                    #     .....

                    criterion = cross_entropy_for_onehot
                    pred_a = net_a(gt_data_a)
                    pred_b = net_b(gt_data_b)
                    ######################## defense start ############################
                    ######################## defense1: trainable layer ############################
                    if self.apply_trainable_layer:
                        active_aggregate_model = ActivePartyWithTrainableLayer(input_dim=num_classes * 2, output_dim=num_classes)
                        dummy_active_aggregate_model = ActivePartyWithTrainableLayer(input_dim=num_classes * 2, output_dim=num_classes)
                    else:
                        active_aggregate_model = ActivePartyWithoutTrainableLayer()
                        dummy_active_aggregate_model = ActivePartyWithoutTrainableLayer()
                    pred = active_aggregate_model(pred_a, pred_b)
                    loss = criterion(pred, gt_onehot_label)
                    ######################## defense2: dp ############################
                    pred_a_gradients = torch.autograd.grad(loss, pred_a, retain_graph=True)
                    pred_a_gradients_clone = pred_a_gradients[0].detach().clone()
                    if self.apply_laplace and self.dp_strength != 0 or self.apply_gaussian and self.dp_strength != 0:
                        location = 0.0
                        threshold = 0.2  # 1e9
                        if self.apply_laplace:
                            with torch.no_grad():
                                scale = self.dp_strength
                                # clip 2-norm per sample
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),threshold + 1e-6).clamp(min=1.0)
                                # add laplace noise
                                dist_a = torch.distributions.laplace.Laplace(location, scale)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                           dist_a.sample(pred_a_gradients_clone.shape).to(self.device)
                        elif self.apply_gaussian:
                            with torch.no_grad():
                                scale = self.dp_strength
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
                                                           threshold + 1e-6).clamp(min=1.0)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                                       torch.normal(location, scale, pred_a_gradients_clone.shape).to(self.device)
                    ######################## defense3: gradient sparsification ############################
                    elif self.apply_grad_spar and self.grad_spars != 0:
                        with torch.no_grad():
                            percent = self.grad_spars / 100.0
                            up_thr = torch.quantile(torch.abs(pred_a_gradients_clone), percent)
                            active_up_gradients_res = torch.where(
                                torch.abs(pred_a_gradients_clone).double() < up_thr.item(),
                                pred_a_gradients_clone.double(), float(0.)).to(self.device)
                            pred_a_gradients_clone = pred_a_gradients_clone - active_up_gradients_res
                    ######################## defense4: marvell ############################
                    elif self.apply_marvell and self.marvell_s != 0 and num_classes == 2:
                        # for marvell, change label to [0,1]
                        marvell_y = []
                        for i in range(len(gt_label)):
                            marvell_y.append(int(gt_label[i][1]))
                        marvell_y = np.array(marvell_y)
                        shared_var.batch_y = np.asarray(marvell_y)
                        logdir = 'marvell_logs/dlg_task/{}_logs/{}'.format(self.dataset, time.strftime("%Y%m%d-%H%M%S"))
                        writer = tf.summary.create_file_writer(logdir)
                        shared_var.writer = writer
                        with torch.no_grad():
                            pred_a_gradients_clone = KL_gradient_perturb(pred_a_gradients_clone, classes, self.marvell_s)
                            pred_a_gradients_clone = pred_a_gradients_clone.to(self.device)
                    ######################## defense5: ppdl, GradientCompression, laplace_noise, DiscreteSGD ############################
                    elif self.apply_ppdl:
                        dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[pred_a_gradients_clone], theta_u=self.ppdl_theta_u, gamma=0.001, tau=0.0001)
                    elif self.apply_gc:
                        tensor_pruner = TensorPruner(zip_percent=self.gc_preserved_percent)
                        tensor_pruner.update_thresh_hold(pred_a_gradients_clone)
                        pred_a_gradients_clone = tensor_pruner.prune_tensor(pred_a_gradients_clone)
                    elif self.apply_lap_noise:
                        dp = DPLaplacianNoiseApplyer(beta=self.noise_scale)
                        pred_a_gradients_clone = dp.laplace_mech(pred_a_gradients_clone)
                    elif self.apply_discrete_gradients:
                        pred_a_gradients_clone = multistep_gradient(pred_a_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
                    original_dy_dx = torch.autograd.grad(pred_a, net_a.parameters(), grad_outputs=pred_a_gradients_clone)
                    ######################## defense end ############################

                    dummy_pred_b = torch.randn(pred_b.size()).to(self.device).requires_grad_(True)
                    dummy_label = torch.randn(gt_onehot_label.size()).to(self.device).requires_grad_(True)

                    if self.apply_trainable_layer:
                        optimizer = torch.optim.Adam([dummy_pred_b, dummy_label] + list(dummy_active_aggregate_model.parameters()), lr=self.lr)
                    else:
                        optimizer = torch.optim.Adam([dummy_pred_b, dummy_label], lr=self.lr)

                    for iters in range(1, self.epochs + 1):
                        def closure():
                            optimizer.zero_grad()
                            dummy_pred = dummy_active_aggregate_model(net_a(gt_data_a), dummy_pred_b)

                            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                            dummy_dy_dx_a = torch.autograd.grad(dummy_loss, net_a.parameters(), create_graph=True)
                            grad_diff = 0
                            for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        rec_rate = self.calc_label_recovery_rate(dummy_label, gt_label)
                        optimizer.step(closure)
                        if self.early_stop == True:
                            if closure().item() < self.early_stop_param:
                                break

                    rec_rate = self.calc_label_recovery_rate(dummy_label, gt_label)
                    recovery_rate_history.append(rec_rate)
                    end_time = time.time()
                    # output the rec_info of this exp
                    if self.apply_laplace or self.apply_gaussian:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,dp_strength=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.dp_strength,rec_rate, end_time - start_time))
                    elif self.apply_grad_spar:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,grad_spars=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.grad_spars,rec_rate, end_time - start_time))
                    elif self.apply_encoder or self.apply_adversarial_encoder:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ae_lambda=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, float(self.ae_lambda), rec_rate, end_time - start_time))
                    elif self.apply_marvell:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,marvel_s=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.marvell_s,rec_rate, end_time - start_time))
                    elif self.apply_ppdl:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ppdl_theta_u=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.ppdl_theta_u,rec_rate, end_time - start_time))
                    elif self.apply_gc:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,gc_preserved_percent=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.gc_preserved_percent,rec_rate, end_time - start_time))
                    elif self.apply_lap_noise:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,noise_scale=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.noise_scale,rec_rate, end_time - start_time))
                    elif self.apply_discrete_gradients:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,marvel_s=%lf,discrete_gradients_bins=%d,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.discrete_gradients_bins,rec_rate, end_time - start_time))
                    else:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, rec_rate, end_time - start_time))
                avg_rec_rate = np.mean(recovery_rate_history)
                if self.apply_laplace or self.apply_gaussian:
                    exp_result = str(self.dp_strength) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_grad_spar:
                    exp_result = str(self.grad_spars) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_encoder or self.apply_adversarial_encoder:
                    exp_result = str(self.ae_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_marvell:
                    exp_result = str(self.marvell_s) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_ppdl:
                    exp_result = str(self.ppdl_theta_u) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_gc:
                    exp_result = str(self.gc_preserved_percent) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_lap_noise:
                    exp_result = str(self.noise_scale) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_discrete_gradients:
                    exp_result = str(self.discrete_gradients_bins) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                else:
                    exp_result = f"bs|num_class|recovery_rate,%d|%d|%lf|%s|%lf" % (batch_size, num_classes, avg_rec_rate, str(recovery_rate_history), np.max(recovery_rate_history))

                append_exp_res(self.exp_res_path, exp_result)
                print(exp_result)

if __name__ == '__main__':
    pass