import torch.cuda

from models.autoencoder import AutoEncoder
from vfl_dlg_no_defense import *
from utils import *

models_dict = {"mnist": 'MLP2',
               "cifar10": 'resnet18',
               "cifar100": 'resnet18',
               "nuswide": 'MLP2',
               "classifier": None}
epochs_dict = {"mnist": 10000,
              "cifar10": 200,
              "cifar100": 200,
              "nuswide": 20000}

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str, help='the dataset which the experiment is based on')
    parser.add_argument('--num_exp', default=10, type=int , help='the number of random experiments')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--early_stop', default=False, type=bool, help='whether to use early stop')
    parser.add_argument('--early_stop_param', default=0.0001, type=float, help='stop train when the loss <= early_stop_param')
    parser.add_argument('--seed', default=100, type=int, help='random seed')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--acc_top_k', default=5, type=int, help='')
    # defense
    parser.add_argument('--apply_trainable_layer', default=False, type=bool)
    parser.add_argument('--apply_laplace', default=False, type=bool, help='whether to use dp-laplace')
    parser.add_argument('--apply_gaussian', default=False, type=bool, help='whether to use dp-gaussian')
    parser.add_argument('--dp_strength', default=0, type=float, help='the parameter of dp defense')
    parser.add_argument('--apply_grad_spar', default=False, type=bool, help='whether to use gradient sparsification')
    parser.add_argument('--grad_spars', default=0, type=float, help='the parameter of gradient sparsification')
    parser.add_argument('--apply_encoder', default=False, type=bool, help='whether to use CoAE')
    parser.add_argument('--apply_random_encoder', default=False, type=bool, help='whether to use random CoAE')
    parser.add_argument('--apply_adversarial_encoder', default=False, type=bool, help='whether to use AAE')

    parser.add_argument('--apply_certify', default=0, type=int, help='whether to use certify')
    parser.add_argument('--certify_M', default=1000, type=int, help='number of voters in CertifyFL')
    parser.add_argument('--certify_start_epoch', default=0, type=int, help='number of epoch that start certify process')

    parser.add_argument('--apply_marvell', default=False, type=bool, help='whether to use marvell(optimal gaussian noise)')
    parser.add_argument('--marvell_s', default=1, type=int, help='scaler of bound in MARVELL')

    # defense methods given in 
    parser.add_argument('--apply_ppdl', help='turn_on_privacy_preserving_deep_learning', type=bool, default=False)
    parser.add_argument('--ppdl_theta_u', help='theta-u parameter for defense privacy-preserving deep learning', type=float, default=0.5)
    parser.add_argument('--apply_gc', help='turn_on_gradient_compression', type=bool, default=False)
    parser.add_argument('--gc_preserved_percent', help='preserved-percent parameter for defense gradient compression', type=float, default=0.1)
    parser.add_argument('--apply_lap_noise', help='turn_on_lap_noise', type=bool, default=False)
    parser.add_argument('--noise_scale', help='noise-scale parameter for defense noisy gradients', type=float, default=1e-3)
    parser.add_argument('--apply_discrete_gradients', default=False, type=bool, help='whether to use Discrete Gradients')
    parser.add_argument('--discrete_gradients_bins', default=12, type=int, help='number of bins for discrete gradients')
    parser.add_argument('--discrete_gradients_bound', default=3e-4, type=float, help='value of bound for discrete gradients')

    args = parser.parse_args()
    set_seed(args.seed)
    args.model = models_dict[args.dataset]
    args.epochs = epochs_dict[args.dataset]

    if args.device == 'cuda':
        cuda_id = args.gpu
        torch.cuda.set_device(cuda_id)
    print(f'running on cuda{torch.cuda.current_device()}')

    # a series of exps
    if args.dataset == 'cifar100':
        args.dst = datasets.CIFAR100("./dataset/", download=True)
        args.num_class_list = [20]
        # args.num_class_list = [2]
    elif args.dataset == 'cifar10':
        args.dst = datasets.CIFAR10("./dataset/", download=True)
        args.num_class_list = [10]
        # args.num_class_list = [2]
    elif args.dataset == 'mnist':
        args.dst = datasets.MNIST("~/.torch", download=True)
        args.num_class_list = [10]
        # args.num_class_list = [2]
    elif args.dataset == 'nuswide':
        args.dst = None
        args.num_class_list = [5]
        # args.num_class_list = [2]
    args.batch_size_list = [2048]


    args.encoder = None
    args.ae_lambda = None
    args.parameter = 0.0
    args.exp_res_dir = f'exp_result_cifar100/{args.dataset}/'
    if args.apply_ppdl:
        args.exp_res_dir = args.exp_res_dir + 'ppdl/'
        args.parameter = args.ppdl_theta_u
    elif args.apply_gc:
        args.exp_res_dir = args.exp_res_dir + 'gradient_compression/'
        args.parameter = args.gc_preserved_percent
    elif args.apply_lap_noise:
        args.exp_res_dir = args.exp_res_dir + 'laplase_noise_gradient/'
        args.parameter = args.noise_scale
    elif args.apply_discrete_gradients:
        args.exp_res_dir = args.exp_res_dir + 'discrete_gradients/'
        args.parameter = args.discrete_gradients_bins
    if not os.path.exists(args.exp_res_dir):
        os.makedirs(args.exp_res_dir)
    filename = f'dlg_task_rec.txt'
    args.exp_res_path = args.exp_res_dir + filename
    dim = args.num_class_list[0]
    label_leakage = LabelLeakage(args)
    label_leakage.train()
