_config_lag = {
    "height": 64,
    "width": 64,
    "crop": True,
    "channel": 3,
    "data_folder": "/content/lag",
    "target_celeblities_num": 200,
    "weight_decay": 0.0001,
}

_config_lfw = {
    "height": 64,
    "width": 64,
    "channel": 3,
    "crop": False,
    "target_celeblities_num": 200,
    "weight_decay": 0.0001,
}

_config_celeba = {
    "height": 64,
    "width": 64,
    "channel": 3,
    "crop": False,
    "target_celeblities_num": 200,
    "weight_decay": 0.003,
}

_config_facescrub = {
    "height": 64,
    "width": 64,
    "channel": 3,
    "crop": False,
    "target_celeblities_num": 200,
    "blur_strength": 10,
    "weight_decay": 0.003,
}

config_dataset = {
    "LAG": _config_lag,
    "LFW": _config_lfw,
    "CelebA": _config_celeba,
    "FaceScrub": _config_facescrub,
}

_config_dsfl = {
    "aggregation": "ERA",
    "era_temperature": 0.1,
    "epoch_local_training": 2,
    "epoch_local_distillation": 2,
    "epoch_global_distillation": 1,
}
_config_fedmd = {
    "consensus_epoch": 1,
    "revisit_epoch": 1,
    "transfer_epoch_private": 5,
    "transfer_epoch_public": 5,
    "server_training_epoch": 1,
    "use_server_model": True,
}
_config_fedgems = {
    "epoch_client_on_localdataset": 2,
    "epoch_client_on_publicdataset": 2,
    "epoch_server_on_publicdataset": 1,
    "epsilon": 0.75,
}
config_fedkd = {
    "FedGEMS": _config_fedgems,
    "FedMD": _config_fedmd,
    "DSFL": _config_dsfl,
}

config_base = {
    "model_type": "CNN",
    "invmodel_type": "InvCNN",
    "num_communication": 5,
    "batch_size": 64,
    "inv_batch_size": 8,
    "lr": 0.001,
    "num_workers": 1,
    "num_classes": 1000,
    "inv_epoch": 3,
    "inv_lr": 0.00003,
    "loss_type": "mse",
}

config_gradinvattack = {
    "distancename": "cossim",
    "num_iteration": 150,
    "optimize_label": False,
    "tv_reg_coef": 0.01,
    "gradinvattack_kwargs": {"lr": 0.3},
    "optimizername": "Adam",
}
