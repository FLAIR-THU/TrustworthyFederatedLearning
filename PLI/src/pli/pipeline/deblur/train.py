import os

import cv2
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .adversarial_trainer import GANFactory
from .models.losses import get_loss
from .models.models import get_model
from .models.networks import get_nets
from .schedulers import LinearDecay

cv2.setNumThreads(0)


class DeblurTrainer:
    def __init__(self, train: DataLoader):
        self.train_dataset = train
        self.adv_lambda = 0.001
        self.warmup_epochs = 3

    def train(self, output_folder):
        self._init_params()
        for epoch in range(0, 200):
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)
            self._run_epoch(epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            if epoch % 10 == 0:
                torch.save(
                    {"model": self.netG.state_dict()},
                    os.path.join(output_folder, "last_{}.h5".format(epoch)),
                )

    def _run_epoch(self, epoch):
        for i, data in enumerate(self.train_dataset):
            targets, inputs = self.model.get_input(data)
            outputs = self.netG(inputs)
            _ = self._update_d(outputs, targets)
            self.optimizer_G.zero_grad()
            loss_content = self.criterionG(outputs, targets)
            loss_adv = self.adv_trainer.loss_g(outputs, targets)
            loss_G = loss_content + self.adv_lambda * loss_adv
            loss_G.backward()
            self.optimizer_G.step()

            if i > 1000:
                break

        figure = plt.figure()
        figure.add_subplot(1, 3, 1)
        plt.imshow(
            cv2.cvtColor(
                inputs[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                cv2.COLOR_BGR2RGB,
            )
        )
        figure.add_subplot(1, 3, 2)
        plt.imshow(
            cv2.cvtColor(
                targets[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                cv2.COLOR_BGR2RGB,
            )
        )
        figure.add_subplot(1, 3, 3)
        plt.imshow(
            cv2.cvtColor(
                outputs[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                cv2.COLOR_BGR2RGB,
            )
        )
        plt.savefig(f"{epoch}.png")

    def _update_d(self, outputs, targets):
        self.optimizer_D.zero_grad()
        loss_D = self.adv_lambda * self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        optimizer = optim.Adam(params, lr=0.0001)
        return optimizer

    def _get_scheduler(self, optimizer):
        scheduler = LinearDecay(
            optimizer,
            min_lr=0.0000001,
            num_epochs=200,
            start_epoch=50,
        )
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == "no_gan":
            return GANFactory.create_model("NoGAN")
        elif d_name == "patch_gan" or d_name == "multi_scale":
            return GANFactory.create_model("SingleGAN", net_d, criterion_d)
        elif d_name == "double_gan":
            return GANFactory.create_model("DoubleGAN", net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionG, criterionD = get_loss()
        self.netG, netD = get_nets()
        self.netG.cuda()
        self.adv_trainer = self._get_adversarial_trainer("double_gan", netD, criterionD)
        self.model = get_model()
        self.optimizer_G = self._get_optim(
            filter(lambda p: p.requires_grad, self.netG.parameters())
        )
        self.optimizer_D = self._get_optim(self.adv_trainer.get_params())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)
