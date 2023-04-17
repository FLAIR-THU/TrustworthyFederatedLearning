import os
import pickle
import random

import numpy as np
import torch

from ...model.model import get_model_class
from ...utils.dataloader import prepare_dataloaders
from ...utils.fedkd_setup import get_fedkd_api


def confidence_gap_fedkd(
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
        _,
        _,
    ) = prepare_dataloaders(
        dataset_name=dataset,
        client_num=client_num,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        num_classes=num_classes,
        **config_dataset,
    )

    # --- Setup Models --- #
    model_class = get_model_class(model_type)
    input_dim = config_dataset["height"] * config_dataset["width"]
    input_dim = (
        input_dim
        if "channel" not in config_dataset
        else input_dim * config_dataset["channel"]
    )

    def calculate_entropy_from_dataloader(model, dataloader, device):
        entropy_list = []
        for data in dataloader:
            _, x, _ = data
            x = x.to(device)
            y_preds = model(x)
            y_preds = y_preds.detach().cpu()
            y_probs = y_preds.softmax(dim=1)
            y_entropy = (-1 * y_probs * torch.log(y_probs)).sum(dim=1)
            entropy_list.append(y_entropy)
        return torch.cat(entropy_list)

    def create_fn_calculate_entropy(output_dir):
        def calculate_entropy(api):
            torch.save(
                calculate_entropy_from_dataloader(
                    api.server, api.public_dataloader, device
                ),
                os.path.join(output_dir, f"{api.epoch}_server_public.pth"),
            )

            for client_idx in range(api.client_num):
                client = api.clients[client_idx]
                torch.save(
                    calculate_entropy_from_dataloader(
                        client, api.public_dataloader, device
                    ),
                    os.path.join(
                        output_dir, f"{api.epoch}_{client_idx}_client_public.pth"
                    ),
                )
                torch.save(
                    calculate_entropy_from_dataloader(
                        client, api.local_dataloaders[client_idx], device
                    ),
                    os.path.join(
                        output_dir, f"{api.epoch}_{client_idx}_client_local.pth"
                    ),
                )
                torch.save(
                    calculate_entropy_from_dataloader(
                        api.server, api.local_dataloaders[client_idx], device
                    ),
                    os.path.join(
                        output_dir, f"{api.epoch}_{client_idx}_server_local.pth"
                    ),
                )

        return calculate_entropy

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
        custom_action=create_fn_calculate_entropy(output_dir),
        target_celeblities_num=config_dataset["target_celeblities_num"],
    )

    fedkd_result = api.run()
    with open(os.path.join(output_dir, "fedkd_result.pkl"), "wb") as f:
        pickle.dump(fedkd_result, f)
