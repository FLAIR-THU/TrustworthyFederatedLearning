import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn

from ..model.invmodel import get_invmodel_class
from ..utilsdata.utils import NumpyDataset, worker_init_fn


def setup_training_based_inversion(
    attack_type,
    invmodel_type,
    num_classes,
    client_num,
    inv_lr,
    device,
    config_dataset,
    temp_dir,
    ablation_study,
):
    if attack_type == "pli":
        inv_input_dim = num_classes * 2 if ablation_study == 2 else num_classes
    else:
        inv_input_dim = num_classes
    temp_path_list = []
    for i in range(client_num):
        inv = get_invmodel_class(invmodel_type)(
            input_dim=inv_input_dim,
            output_shape=(
                config_dataset["channel"],
                config_dataset["height"],
                config_dataset["width"],
            ),
            channel=config_dataset["channel"],
        ).to(device)
        inv_optimizer = torch.optim.Adam(
            inv.parameters(), lr=inv_lr, weight_decay=0.0001
        )
        state = {
            "model": inv.state_dict(),
            "optimizer": inv_optimizer.state_dict(),
        }
        temp_path = os.path.join(temp_dir, f"client_{i}")
        torch.save(state, temp_path + ".pth")
        temp_path_list.append(temp_path)

    return temp_path_list, inv, inv_optimizer


def setup_tbi_optimizers(dataset_name, config_dataset):
    transforms_list = [transforms.ToTensor()]
    if dataset_name not in ["AT&T", "MNIST"]:
        if "channel" not in config_dataset or config_dataset["channel"] != 3:
            transforms_list.append(transforms.Grayscale())
    if "crop" in config_dataset and config_dataset["crop"]:
        transforms_list.append(
            transforms.CenterCrop(
                (max(config_dataset["height"], config_dataset["width"]))
            )
        )
    else:
        transforms_list.append(
            transforms.Resize((config_dataset["height"], config_dataset["width"]))
        )
    if "channel" not in config_dataset or config_dataset["channel"] == 1:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    inv_transform = transforms.Compose(transforms_list)
    return inv_transform


def setup_tbi_inv_dataloader(
    target_labels,
    is_sensitive_flag,
    api,
    target_client_api_list,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
):
    inv_trainset = NumpyDataset(
        x=api.public_dataloader.dataset.x,
        y=api.public_dataloader.dataset.y,
        transform=inv_transform,
        return_idx=return_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    inv_public_dataloader = torch.utils.data.DataLoader(
        inv_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # --- Receive logits --- #
    public_x_list = []
    y_pred_local_list = []

    for target_client_api in target_client_api_list:
        for data in inv_public_dataloader:
            idx = data[0]
            x = data[1].to(device).detach()
            y_pred_local = torch.softmax(
                target_client_api(x) / inv_tempreature, dim=-1
            ).detach()
            public_x_list.append(x.cpu())
            y_pred_local_list.append(y_pred_local.cpu())

    public_x_tensor = torch.cat(public_x_list)
    y_pred_local_tensor = torch.cat(y_pred_local_list)

    prediction_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(public_x_tensor, y_pred_local_tensor),
        batch_size=inv_batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    return prediction_dataloader


def setup_our_inv_dataloader(
    target_labels,
    is_sensitive_flag,
    api,
    target_client_api_list,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
    only_sensitive=True,
):
    if only_sensitive:
        sensitive_flag = np.where(is_sensitive_flag == 1)[0]
        inv_trainset = NumpyDataset(
            x=api.public_dataloader.dataset.x[sensitive_flag],
            y=api.public_dataloader.dataset.y[sensitive_flag],
            transform=inv_transform,
            return_idx=return_idx,
        )
    else:
        inv_trainset = NumpyDataset(
            x=api.public_dataloader.dataset.x,
            y=api.public_dataloader.dataset.y,
            transform=inv_transform,
            return_idx=return_idx,
        )

    g = torch.Generator()
    g.manual_seed(seed)
    inv_public_dataloader = torch.utils.data.DataLoader(
        inv_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # --- Receive logits --- #
    public_x_list = []
    y_label_list = []
    y_pred_server_list = []
    y_pred_local_list = []
    for target_client_api in target_client_api_list:
        for data in inv_public_dataloader:
            x = data[1].to(device).detach()
            y_label = data[2]
            y_pred_server = torch.softmax(
                api.server(x) / inv_tempreature, dim=-1
            ).detach()
            y_pred_local = torch.softmax(
                target_client_api(x) / inv_tempreature, dim=-1
            ).detach()
            public_x_list.append(x.cpu())
            y_label_list.append(y_label.cpu())
            y_pred_server_list.append(y_pred_server.cpu())
            y_pred_local_list.append(y_pred_local.cpu())

    public_x_tensor = torch.cat(public_x_list)
    y_label_tensor = torch.cat(y_label_list)
    y_pred_server_tensor = torch.cat(y_pred_server_list)
    y_pred_local_tensor = torch.cat(y_pred_local_list)

    prediction_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            public_x_tensor, y_pred_server_tensor, y_pred_local_tensor, y_label_tensor
        ),
        batch_size=inv_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    return prediction_dataloader


def setup_our_inv_dataloader_from_single_client(
    target_labels,
    is_sensitive_flag,
    api,
    target_client_api,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
    only_sensitive=True,
):
    if only_sensitive:
        sensitive_flag = np.where(is_sensitive_flag == 1)[0]
        inv_trainset = NumpyDataset(
            x=api.public_dataloader.dataset.x[sensitive_flag],
            y=api.public_dataloader.dataset.y[sensitive_flag],
            transform=inv_transform,
            return_idx=return_idx,
        )
    else:
        inv_trainset = NumpyDataset(
            x=api.public_dataloader.dataset.x,
            y=api.public_dataloader.dataset.y,
            transform=inv_transform,
            return_idx=return_idx,
        )

    g = torch.Generator()
    g.manual_seed(seed)
    inv_public_dataloader = torch.utils.data.DataLoader(
        inv_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # --- Receive logits --- #
    public_x_list = []
    y_label_list = []
    y_pred_server_list = []
    y_pred_local_list = []
    for data in inv_public_dataloader:
        x = data[1].to(device).detach()
        y_label = data[2]
        y_pred_server = torch.softmax(api.server(x) / inv_tempreature, dim=-1).detach()
        y_pred_local = torch.softmax(
            target_client_api(x) / inv_tempreature, dim=-1
        ).detach()
        public_x_list.append(x.cpu())
        y_label_list.append(y_label.cpu())
        y_pred_server_list.append(y_pred_server.cpu())
        y_pred_local_list.append(y_pred_local.cpu())

    public_x_tensor = torch.cat(public_x_list)
    y_label_tensor = torch.cat(y_label_list)
    y_pred_server_tensor = torch.cat(y_pred_server_list)
    y_pred_local_tensor = torch.cat(y_pred_local_list)

    prediction_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            public_x_tensor, y_pred_server_tensor, y_pred_local_tensor, y_label_tensor
        ),
        batch_size=inv_batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    return prediction_dataloader
