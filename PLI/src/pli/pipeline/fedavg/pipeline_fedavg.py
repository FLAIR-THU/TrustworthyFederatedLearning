import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from aijack.collaborative import FedAvgClient, FedAvgServer
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

from ...attack.gradinv.gradientinversion import GradientInversionAttackManager
from ...model.model import get_model_class
from ...utils.dataloader import prepare_dataloaders
from ...utils.utils_data import extract_transformd_dataset_from_dataloader


def attack_fedavg(
    model_type="OneCNN",
    dataset="LAG",
    client_num=2,
    batch_size=4,
    lr=0.01,
    num_classes=20,
    num_communication=3,
    seed=42,
    num_workers=2,
    config_dataset=None,
    config_gradinvattack=None,
    output_dir="",
):
    # --- Fix seed --- #
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # --- Setup device --- #
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    # --- Setup DataLoaders --- #
    (_, local_dataloaders, _, local_identities, _) = prepare_dataloaders(
        dataset_name=dataset,
        client_num=client_num,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        num_classes=num_classes,
        **config_dataset,
    )

    label2newlabel = {
        la: i for i, la in enumerate(np.unique(sum(local_identities, [])))
    }
    for local_dataloader in local_dataloaders:
        for i in range(local_dataloader.dataset.y.shape[0]):
            local_dataloader.dataset.y[i] = label2newlabel[
                local_dataloader.dataset.y[i]
            ]

    local_dataset_nums = [
        dataloader.dataset.x.shape[0] for dataloader in local_dataloaders
    ]

    # --- Setup loss function --- #
    criterion = nn.CrossEntropyLoss()

    # --- Setup Models --- #
    model_class = get_model_class(model_type)
    input_dim = config_dataset["height"] * config_dataset["width"]
    input_dim = (
        input_dim
        if "channel" not in config_dataset
        else input_dim * config_dataset["channel"]
    )

    # --- Setup FedAVG ---#
    # ------ client-side ------ #
    clients = [
        FedAvgClient(
            model_class(
                input_dim=input_dim,
                output_dim=config_dataset["target_celeblities_num"],
                channel=config_dataset["channel"],
            ).to(device),
            user_id=i,
            lr=lr,
        ).to(device)
        for i in range(client_num)
    ]
    client_optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

    # ------ server-side ------- #
    server_model = model_class(
        input_dim=input_dim,
        output_dim=config_dataset["target_celeblities_num"],
        channel=config_dataset["channel"],
    ).to(device)
    manager = GradientInversionAttackManager(
        (config_dataset["channel"], config_dataset["height"], config_dataset["width"]),
        device=device,
        clamp_range=(-1, 1),
        **config_gradinvattack,
    )
    FedAvgServer_GradInvAttack = manager.attach(FedAvgServer)
    server = FedAvgServer_GradInvAttack(clients, server_model, lr=lr)

    reconstructed_xs = [None for _ in range(client_num)]
    reconstructed_ys = [None for _ in range(client_num)]

    for com in range(num_communication):
        for client_idx in range(client_num):
            client = clients[client_idx]
            trainloader = local_dataloaders[client_idx]
            optimizer = client_optimizers[client_idx]

            for i in range(2):
                running_loss = 0.0
                for _, data in enumerate(trainloader, 0):
                    _, inputs, labels = data
                    inputs = inputs.to(device)
                    inputs.requires_grad = True
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    client.zero_grad()

                    outputs = client(inputs)
                    loss = criterion(outputs, labels.to(torch.int64))

                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()
                print(
                    f"communication {com}, epoch {i}: client-{client_idx+1}",
                    running_loss / local_dataset_nums[client_idx],
                )

        server.receive(use_gradients=True)

        for client_idx in range(client_num):
            print(f"attack client_idx={client_idx}")
            print(
                f"batch size is {int(config_dataset['target_celeblities_num'] / client_num)}"
            )
            server.change_target_client_id(client_idx)
            result = server.attack(
                batch_size=int(config_dataset["target_celeblities_num"] / client_num),
                init_x=reconstructed_xs[client_idx],
                return_best=False,
            )
            reconstructed_xs[client_idx] = result[0]
            reconstructed_ys[client_idx] = result[1]

        server.update(use_gradients=True)
        server.distribtue()

        print(f"communication {com} is done")

    transformed_ldl = [
        extract_transformd_dataset_from_dataloader(ldl, return_idx=True)
        for ldl in local_dataloaders
    ]
    private_dataset_transformed = torch.concat([tldl[0] for tldl in transformed_ldl])
    private_dataset_label = torch.concat([tldl[1] for tldl in transformed_ldl])
    suc_num = 0
    best_ssim_list = []

    for j in range(client_num):
        local_identity = np.unique(local_dataloaders[j].dataset.y)
        fake_labels = torch.Tensor(local_identity).to(torch.int64).to(device)

        for x_rec, label in zip(reconstructed_xs[j], reconstructed_ys[j]):
            temp_rec_img = x_rec.detach().cpu().numpy().transpose(1, 2, 0)
            temp_ssim_list = [
                ssim(
                    temp_rec_img,
                    torch.mean(
                        private_dataset_transformed[
                            private_dataset_label == temp_label
                        ],
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0),
                    data_range=2.0,
                    multichannel=True,
                )
                for temp_label in torch.unique(private_dataset_label)
            ]

            suc_num += label.item() == np.nanargmax(temp_ssim_list)
            best_ssim_list.append(np.nanmax(temp_ssim_list))

        fig = plt.figure(figsize=(18, 8))
        img = torchvision.utils.make_grid(reconstructed_xs[j] * 0.5 + 0.5, nrow=12)
        fig.add_subplot(1, 2, 1)
        plt.imshow(
            cv2.cvtColor(img.cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
        )
        plt.axis("off")

        img = torchvision.utils.make_grid(
            torch.stack(
                [
                    torch.Tensor(
                        cv2.resize(
                            np.mean(
                                local_dataloaders[j].dataset.x[
                                    local_dataloaders[j].dataset.y == la.item()
                                ],
                                axis=0,
                            ),
                            dsize=(64, 64),
                        ).transpose(2, 0, 1)
                        / 255
                    )
                    for la in fake_labels
                ]
            ),
            nrow=12,
        )
        fig.add_subplot(1, 2, 2)
        plt.imshow(
            cv2.cvtColor(img.cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
        )
        plt.axis("off")
        plt.savefig(f"{j}.png")
        plt.show()

    for i, (xs, ys) in enumerate(zip(reconstructed_xs, reconstructed_ys)):
        torch.save(xs.detach(), f"{output_dir}/{i}_xs.pth")
        torch.save(ys.detach(), f"{output_dir}/{i}_ys.pth")

    torch.save(
        private_dataset_transformed.detach().cpu(),
        f"{output_dir}/private_dataset_transformed.pth",
    )
    torch.save(
        private_dataset_label.detach().cpu(),
        f"{output_dir}/private_dataset_label.pth",
    )

    return {
        "suc_num": suc_num,
        "ssim_mean": np.mean(best_ssim_list),
        "ssim_std": np.std(best_ssim_list),
    }
