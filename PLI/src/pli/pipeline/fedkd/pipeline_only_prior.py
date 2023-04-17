import math
import os
import random

import cv2
import numpy as np
import torch

from ...model.cycle_gan_model import CycleGANModel
from ...model.networks import define_G
from ...utils.dataloader import prepare_dataloaders
from ..evaluation.evaluation import evaluation_full
from .options import BaseOptions


def attack_prior(
    fedkd_type="FedGEMS",
    model_type="LM",
    invmodel_type="InvCNN",
    attack_type="pli",
    loss_type="mse",
    dataset="AT&T",
    client_num=2,
    batch_size=4,
    inv_batch_size=1,
    lr=0.01,
    num_classes=20,
    num_communication=5,
    seed=42,
    num_workers=2,
    inv_epoch=10,
    inv_lr=0.003,
    inv_tempreature=1.0,
    alpha=3.0,
    gamma=0.1,
    ablation_study=0,
    config_fedkd=None,
    config_dataset=None,
    output_dir="",
    temp_dir="./",
    model_path="./",
    only_sensitive=True,
    use_multi_models=False,
):
    # --- Fix seed --- #
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except:
        print("torch.use_deterministic_algorithms is not available")

    try:
        torch.backends.cudnn.benchmark = False
    except:
        print("torch.backends.cudnn.benchmark is not available")

    # --- Setup device --- #
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print("device is ", device)

    # --- Setup DataLoaders --- #
    (
        public_train_dataloader,
        local_train_dataloaders,
        test_dataloader,
        local_identities,
        is_sensitive_flag,
    ) = prepare_dataloaders(
        dataset_name=dataset,
        client_num=client_num,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        num_classes=num_classes,
        **config_dataset,
    )

    output_dim = (
        num_classes
        if fedkd_type != "DSFL"
        else config_dataset["target_celeblities_num"]
    )

    # --- Setup Models --- #
    input_dim = config_dataset["height"] * config_dataset["width"]
    input_dim = (
        input_dim
        if "channel" not in config_dataset
        else input_dim * config_dataset["channel"]
    )

    if fedkd_type == "DSFL":
        id2label = {la: i for i, la in enumerate(np.unique(sum(local_identities, [])))}
    else:
        id2label = {la: la for la in sum(local_identities, [])}

    nonsensitive_idxs = np.where(is_sensitive_flag == 0)[0]
    x_pub_nonsensitive = torch.stack(
        [
            public_train_dataloader.dataset.transform(
                public_train_dataloader.dataset.x[nidx]
            )
            for nidx in nonsensitive_idxs
        ]
    )
    y_pub_nonsensitive = torch.Tensor(
        public_train_dataloader.dataset.y[nonsensitive_idxs]
    )

    prior = torch.zeros(
        (
            output_dim,
            config_dataset["channel"],
            config_dataset["height"],
            config_dataset["width"],
        )
    )

    target_labels = sum(
        [[id2label[la] for la in temp_list] for temp_list in local_identities], []
    )
    print(target_labels)

    if fedkd_type != "DSFL":
        if dataset == "FaceScrub":
            model = define_G(
                3,
                3,
                64,
                "resnet_3blocks",
                norm="instance",
                use_dropout=False,
                init_type="normal",
                init_gain=0.02,
                gpu_ids=[0],
            )
            model.load_state_dict(
                torch.load(os.path.join(model_path, "last_200.h5"))["model"]
            )
            model.eval()

            for lab in range(output_dim):
                lab_idxs = torch.where(y_pub_nonsensitive == lab)[0]
                lab_idxs_size = lab_idxs.shape[0]
                if lab_idxs_size == 0:
                    continue
                for batch_pos in np.array_split(
                    list(range(lab_idxs_size)), math.ceil(lab_idxs_size / 8)
                ):
                    prior[lab] += (
                        model(x_pub_nonsensitive[lab_idxs[batch_pos]].to(device))
                        .detach()
                        .cpu()
                        .sum(dim=0)
                        / lab_idxs_size
                    )

        else:
            opt = BaseOptions()
            opt.checkpoints_dir = output_dir

            model = CycleGANModel(opt)
            model.setup(opt)
            model.load_networks(50, model_path)
            model.netG_A.eval()

            for lab in target_labels:
                lab_idxs = torch.where(y_pub_nonsensitive == lab)[0]
                lab_idxs_size = lab_idxs.shape[0]
                if lab_idxs_size == 0:
                    continue
                for batch_pos in np.array_split(
                    list(range(lab_idxs_size)), math.ceil(lab_idxs_size / 8)
                ):
                    prior[lab] += (
                        model.netG_A(x_pub_nonsensitive[lab_idxs[batch_pos]].to(device))
                        .detach()
                        .cpu()
                        .sum(dim=0)
                        / lab_idxs_size
                    )

    else:
        sensitive_idxs = np.where(is_sensitive_flag == 1)[0]
        x_pub_sensitive = torch.stack(
            [
                public_train_dataloader.dataset.transform(
                    public_train_dataloader.dataset.x[sidx]
                )
                for sidx in sensitive_idxs
            ]
        )
        prior = torch.zeros(
            (
                output_dim,
                config_dataset["channel"],
                config_dataset["height"],
                config_dataset["width"],
            )
        )
        for lab in target_labels:
            prior[lab] = x_pub_sensitive.mean(dim=0)

    for label in target_labels:
        np_img = prior[label].detach().cpu().numpy()
        np.save(
            os.path.join(
                output_dir,
                f"0_{label}_prior",
            ),
            np_img,
        )

    result = evaluation_full(
        client_num,
        num_classes,
        public_train_dataloader,
        local_train_dataloaders,
        local_identities,
        id2label,
        attack_type,
        output_dir,
        epoch=0,
    )

    return result
