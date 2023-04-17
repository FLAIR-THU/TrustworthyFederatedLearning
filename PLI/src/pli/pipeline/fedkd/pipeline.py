import math
import os
import pickle
import random

import numpy as np
import torch

from ...attack.tbi_train import (
    get_our_inv_train_func,
    get_our_inv_train_func_with_multi_models,
    get_tbi_inv_train_func,
)
from ...model.cycle_gan_model import CycleGANModel
from ...model.model import get_model_class
from ...model.networks import define_G
from ...utils.dataloader import prepare_dataloaders
from ...utils.fedkd_setup import get_fedkd_api
from ...utils.loss import SSIMLoss
from ...utils.tbi_setup import setup_tbi_optimizers, setup_training_based_inversion
from ..evaluation.evaluation import evaluation_full, evaluation_full_multi_models
from .options import BaseOptions


def attack_fedkd(
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

    return_idx = True

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
    model_class = get_model_class(model_type)
    input_dim = config_dataset["height"] * config_dataset["width"]
    input_dim = (
        input_dim
        if "channel" not in config_dataset
        else input_dim * config_dataset["channel"]
    )

    # --- Setup Optimizers --- #
    (inv_path_list, inv, inv_optimizer) = setup_training_based_inversion(
        attack_type,
        invmodel_type,
        output_dim,
        client_num,
        inv_lr,
        device,
        config_dataset,
        temp_dir,
        ablation_study,
    )

    # --- Setup transformers --- #
    inv_transform = setup_tbi_optimizers(dataset, config_dataset)

    # --- Setup loss function --- #
    if loss_type == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_type == "ssim":
        criterion = SSIMLoss()
    else:
        raise NotImplementedError(
            f"{loss_type} is not supported. We currently support `mse` or `ssim`."
        )

    if fedkd_type == "DSFL":
        id2label = {la: i for i, la in enumerate(np.unique(sum(local_identities, [])))}
    else:
        id2label = {la: la for la in sum(local_identities, [])}

    if attack_type == "pli":
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

        if gamma != 0.0:
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
                                model(
                                    x_pub_nonsensitive[lab_idxs[batch_pos]].to(device)
                                )
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

                    for lab in range(output_dim):
                        lab_idxs = torch.where(y_pub_nonsensitive == lab)[0]
                        lab_idxs_size = lab_idxs.shape[0]
                        if lab_idxs_size == 0:
                            continue
                        for batch_pos in np.array_split(
                            list(range(lab_idxs_size)), math.ceil(lab_idxs_size / 8)
                        ):
                            prior[lab] += (
                                model.netG_A(
                                    x_pub_nonsensitive[lab_idxs[batch_pos]].to(device)
                                )
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
                for lab in range(output_dim):
                    prior[lab] = x_pub_sensitive.mean(dim=0)

            torch.save(prior, os.path.join(output_dir, "prior.pth"))
        else:
            prior = None

        if not use_multi_models:
            inv_train = get_our_inv_train_func(
                client_num,
                is_sensitive_flag,
                local_identities,
                inv_transform,
                return_idx,
                seed,
                batch_size,
                num_workers,
                device,
                inv_tempreature,
                inv_batch_size,
                inv_epoch,
                inv,
                inv_optimizer,
                prior,
                criterion,
                output_dim,
                attack_type,
                id2label,
                output_dir,
                ablation_study,
                alpha,
                gamma=gamma,
                only_sensitive=only_sensitive,
            )
        else:
            inv_train = get_our_inv_train_func_with_multi_models(
                client_num,
                is_sensitive_flag,
                local_identities,
                inv_transform,
                return_idx,
                seed,
                batch_size,
                num_workers,
                device,
                inv_path_list,
                inv_tempreature,
                inv_batch_size,
                inv_epoch,
                inv,
                inv_optimizer,
                prior,
                criterion,
                output_dim,
                attack_type,
                id2label,
                output_dir,
                ablation_study,
                alpha,
                gamma=gamma,
                only_sensitive=only_sensitive,
            )

    elif attack_type == "tbi":
        inv_train = get_tbi_inv_train_func(
            client_num,
            local_identities,
            inv_transform,
            return_idx,
            seed,
            batch_size,
            num_workers,
            device,
            inv_tempreature,
            inv_batch_size,
            inv_epoch,
            inv,
            inv_optimizer,
            criterion,
            output_dir,
            attack_type,
            output_dim,
            id2label,
        )
    else:
        raise NotImplementedError(f"{attack_type} is not supported")

    # --- Run FedKD --- #
    api = get_fedkd_api(
        fedkd_type,
        model_class,
        public_train_dataloader,
        local_train_dataloaders,
        test_dataloader,
        num_classes,
        client_num,
        config_dataset["channel"],
        lr,
        num_communication,
        input_dim,
        device,
        config_fedkd,
        custom_action=inv_train,
        target_celeblities_num=config_dataset["target_celeblities_num"],
    )

    fedkd_result = api.run()
    with open(os.path.join(output_dir, "fedkd_result.pkl"), "wb") as f:
        pickle.dump(fedkd_result, f)

    # --- Attack --- #

    if not use_multi_models:
        result = evaluation_full(
            client_num,
            num_classes,
            public_train_dataloader,
            local_train_dataloaders,
            local_identities,
            id2label,
            attack_type,
            output_dir,
            epoch=num_communication,
        )
    else:
        result = evaluation_full_multi_models(
            client_num,
            num_classes,
            public_train_dataloader,
            local_train_dataloaders,
            local_identities,
            id2label,
            attack_type,
            output_dir,
            epoch=num_communication,
        )

    return result
