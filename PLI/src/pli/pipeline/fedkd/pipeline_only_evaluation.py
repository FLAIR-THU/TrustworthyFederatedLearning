import os
import random

import numpy as np
import torch

from ...utils.dataloader import prepare_dataloaders
from ..evaluation.evaluation import evaluation_full, evaluation_full_multi_models


def evaluation_fedkd(
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

    # --- Setup Models --- #

    if fedkd_type == "DSFL":
        id2label = {la: i for i, la in enumerate(np.unique(sum(local_identities, [])))}
    else:
        id2label = {la: la for la in sum(local_identities, [])}

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
            device=device,
            save_gt=False,
            label_transform=True,
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
            device=device,
            save_gt=False,
            label_transform=True,
        )

    return result
