import argparse
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from ...model.cycle_gan_model import CycleGANModel
from ...utils.inv_dataloader import prepare_inv_dataloaders
from ...utils.loss import SSIMLoss
from ..deblur.train import DeblurTrainer
from .options import BaseOptions


def unloader(img):
    image = img.clone()
    image = image.cpu()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)
    return transforms.ToPILImage()(image)


def ae_attack_fedkd(
    dataset="AT&T",
    client_num=2,
    batch_size=4,
    num_classes=20,
    inv_lr=0.00003,
    seed=42,
    num_workers=2,
    loss_type="mse",
    config_dataset=None,
    output_dir="",
):
    # --- Fix seed --- #
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # --- Setup device --- #
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print("device is ", device)

    # --- Setup DataLoaders --- #

    inv_dataloader = prepare_inv_dataloaders(
        dataset_name=dataset,
        client_num=client_num,
        batch_size=1,
        seed=seed,
        num_workers=num_workers,
        num_classes=num_classes,
        **config_dataset,
    )

    # --- Setup loss function --- #
    if loss_type == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_type == "ssim":
        criterion = SSIMLoss()
    else:
        raise NotImplementedError(
            f"{loss_type} is not supported. We currently support `mse` or `ssim`."
        )

    opt = BaseOptions()
    opt.checkpoints_dir = output_dir

    if dataset == "FaceScrub":
        trainer = DeblurTrainer(inv_dataloader)
        trainer.train(output_dir)
    else:
        model = CycleGANModel(opt)
        model.setup(opt)

        for epoch in range(1, opt.n_epochs + 1):
            model.update_learning_rate()
            for data in inv_dataloader:
                x1 = data[1].to(device)
                x2 = data[2].to(device)

                model.set_input(
                    {"A": data[1], "B": data[2]}
                )  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()

            x3 = model.netG_A(x1[[0]])

            figure = plt.figure()
            figure.add_subplot(1, 3, 1)
            plt.imshow(
                cv2.cvtColor(
                    x1[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                    cv2.COLOR_BGR2RGB,
                )
            )
            figure.add_subplot(1, 3, 2)
            plt.imshow(
                cv2.cvtColor(
                    x2[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                    cv2.COLOR_BGR2RGB,
                )
            )
            figure.add_subplot(1, 3, 3)
            plt.imshow(
                cv2.cvtColor(
                    x3[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                    cv2.COLOR_BGR2RGB,
                )
            )
            plt.savefig(f"{epoch}.png")

            if epoch % 10 == 0:
                model.save_networks(epoch)
