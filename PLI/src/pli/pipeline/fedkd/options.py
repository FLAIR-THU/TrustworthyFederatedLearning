class BaseOptions:
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.direction = "AtoB"
        self.model = "cycle_gan"
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netD = "basic"
        self.netG = "resnet_3blocks"
        self.n_layers_D = 3
        self.norm = "instance"
        self.init_type = "normal"
        self.init_gain = 0.02
        self.no_dropout = True
        self.n_epochs = 100
        self.n_epochs_decay = 50
        self.beta1 = 0.5
        self.lr = 0.0002
        self.gan_mode = "lsgan"
        self.pool_size = 50
        self.lr_policy = "linear"
        self.lr_decay_iters = 50
        self.checkpoints_dir = "./"
        self.gpu_ids = [0]

        self.epoch_count = 1

        self.load_iter = 50
        self.continue_train = False

        self.isTrain = True

        self.lambda_A = 3.0
        self.lambda_B = 3.0
        self.lambda_identity = 0.5

        self.verbose = 2

        self.initialized = True
